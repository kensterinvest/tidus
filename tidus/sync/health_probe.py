"""Health probe — periodic latency and availability checks per enabled model.

Runs every 5 minutes (configurable via policies.yaml). For each enabled model:
  1. Sends a minimal test prompt via the model's adapter
  2. Records latency and success/failure
  3. After `failure_threshold` consecutive failures → sets model enabled=False
  4. Updates latency_p50_ms (rolling P50 over last 20 probes)

The probe uses the registered adapter for each vendor. Models whose vendor
has no registered adapter (not yet built) are skipped gracefully.

Example:
    probe = HealthProbe(registry, policy)
    await probe.run_once()
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from statistics import median

import structlog

from tidus.router.registry import ModelRegistry
from tidus.utils.yaml_loader import load_yaml

log = structlog.get_logger(__name__)

_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]
_LATENCY_WINDOW = 20  # rolling window for P50


class HealthProbe:
    """Runs health checks against all enabled models in the registry."""

    def __init__(self, registry: ModelRegistry, policies_path: str = "config/policies.yaml") -> None:
        self._registry = registry
        self._policies_path = policies_path
        self._consecutive_failures: dict[str, int] = defaultdict(int)
        self._latency_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=_LATENCY_WINDOW))

    async def run_once(self) -> dict[str, bool]:
        """Probe all enabled models once. Returns {model_id: is_healthy}."""
        raw = load_yaml(self._policies_path)
        failure_threshold = raw.get("health", {}).get("failure_threshold", 3)

        results: dict[str, bool] = {}

        try:
            from tidus.adapters.adapter_factory import get_adapter
        except ImportError:
            log.warning("health_probe_skipped", reason="adapter_factory not available")
            return results

        for spec in self._registry.list_enabled():
            try:
                adapter = get_adapter(spec.vendor)
            except KeyError:
                log.debug("health_probe_no_adapter", model_id=spec.model_id, vendor=spec.vendor)
                continue

            start = time.monotonic()
            try:

                class _MinimalTask:
                    messages = _PROBE_MESSAGES
                    estimated_output_tokens = 5

                healthy = await adapter.health_check(spec.model_id)
                latency_ms = (time.monotonic() - start) * 1000
            except Exception as exc:
                healthy = False
                latency_ms = (time.monotonic() - start) * 1000
                log.warning("health_probe_error", model_id=spec.model_id, error=str(exc))

            results[spec.model_id] = healthy

            if healthy:
                self._consecutive_failures[spec.model_id] = 0
                self._latency_history[spec.model_id].append(latency_ms)
                history = list(self._latency_history[spec.model_id])
                if history:
                    new_p50 = int(median(history))
                    self._registry.update_latency(spec.model_id, new_p50)
                log.debug(
                    "health_probe_ok",
                    model_id=spec.model_id,
                    latency_ms=round(latency_ms, 1),
                )
            else:
                self._consecutive_failures[spec.model_id] += 1
                failures = self._consecutive_failures[spec.model_id]
                log.warning(
                    "health_probe_fail",
                    model_id=spec.model_id,
                    consecutive_failures=failures,
                    threshold=failure_threshold,
                )
                if failures >= failure_threshold:
                    self._registry.set_enabled(spec.model_id, False)
                    log.error(
                        "model_auto_disabled",
                        model_id=spec.model_id,
                        reason=f"{failures} consecutive health probe failures",
                    )

        return results
