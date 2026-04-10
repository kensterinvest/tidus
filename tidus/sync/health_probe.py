"""Health probe — periodic latency and availability checks per enabled model.

Runs every 5 minutes (configurable via policies.yaml). Classifies each enabled
model into one of three probe tiers to balance accuracy against API cost:

  Tier A — always probe live (health_check call):
    Models with consecutive_failures > 0 or with a warning-level drift event.

  Tier B — synthetic-first, escalate to live only if synthetic fails:
    Models with no recent telemetry (not probed in the last 30 minutes).

  Tier C — 10% random sample, synthetic-first:
    Healthy models probed recently. Reduces live API calls on large registries.

"Synthetic probe" = adapter.count_tokens(model_id, ["hi"]) — free tokenization
call that does not invoke the LLM. Only Tier A always starts with a live
health_check().

Results are written to model_telemetry via TelemetryWriter after each probe.
The EffectiveRegistry reads these rows (with staleness classification) to update
the merge layer between restarts.

Example:
    probe = HealthProbe(registry, policies_path, session_factory)
    results = await probe.run_once()
"""

from __future__ import annotations

import random
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from statistics import median

import structlog

from tidus.observability.registry_metrics import PROBE_LIVE_CALLS, PROBE_SYNTHETIC_CALLS
from tidus.sync.telemetry_writer import TelemetryWriter
from tidus.utils.yaml_loader import load_yaml

log = structlog.get_logger(__name__)

_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]
_LATENCY_WINDOW = 20          # rolling window for P50
_TIER_B_WINDOW = timedelta(minutes=30)   # no telemetry in this window → Tier B
_TIER_C_SAMPLE_RATE = 0.10   # fraction of Tier C models to probe each cycle


class HealthProbe:
    """Runs health checks against enabled models with 3-tier sampling."""

    def __init__(
        self,
        registry,
        policies_path: str = "config/policies.yaml",
        session_factory=None,
    ) -> None:
        self._registry = registry
        self._policies_path = policies_path
        self._sf = session_factory
        self._consecutive_failures: dict[str, int] = defaultdict(int)
        # Rolling window resets on restart — the first post-restart probe updates
        # update_latency() with a single sample until the window fills (_LATENCY_WINDOW probes).
        self._latency_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=_LATENCY_WINDOW))

    async def run_once(self) -> dict[str, bool]:
        """Probe models according to tier classification. Returns {model_id: is_healthy}."""
        raw = load_yaml(self._policies_path)
        failure_threshold = raw.get("health", {}).get("failure_threshold", 3)

        results: dict[str, bool] = {}

        try:
            from tidus.adapters.adapter_factory import get_adapter
        except ImportError:
            log.warning("health_probe_skipped", reason="adapter_factory not available")
            return results

        # Load recent telemetry to classify tiers (Tier C = everything else)
        tier_a_ids, tier_b_ids = await self._classify_tiers()

        for spec in self._registry.list_enabled():
            try:
                adapter = get_adapter(spec.vendor)
            except KeyError:
                log.debug("health_probe_no_adapter", model_id=spec.model_id, vendor=spec.vendor)
                continue

            model_id = spec.model_id

            # Determine tier and whether to probe this cycle
            if model_id in tier_a_ids:
                probe_type = "live"
            elif model_id in tier_b_ids:
                probe_type = "synthetic"
            else:
                # Tier C — 10% random sample
                if random.random() > _TIER_C_SAMPLE_RATE:
                    continue
                probe_type = "synthetic"

            start = time.monotonic()
            healthy, actual_probe_type = await self._probe(
                adapter, model_id, probe_type, failure_threshold
            )
            latency_ms = (time.monotonic() - start) * 1000

            results[model_id] = healthy

            if healthy:
                self._consecutive_failures[model_id] = 0
                self._latency_history[model_id].append(latency_ms)
                history = list(self._latency_history[model_id])
                if history:
                    self._registry.update_latency(model_id, int(median(history)))
                log.debug(
                    "health_probe_ok",
                    model_id=model_id,
                    latency_ms=round(latency_ms, 1),
                    probe_type=actual_probe_type,
                )
            else:
                self._consecutive_failures[model_id] += 1
                failures = self._consecutive_failures[model_id]
                log.warning(
                    "health_probe_fail",
                    model_id=model_id,
                    consecutive_failures=failures,
                    threshold=failure_threshold,
                    probe_type=actual_probe_type,
                )
                if failures >= failure_threshold:
                    self._registry.set_enabled(model_id, False)
                    log.error(
                        "model_auto_disabled",
                        model_id=model_id,
                        reason=f"{failures} consecutive health probe failures",
                    )

            # Increment Prometheus probe counters
            result_label = "success" if healthy else "fail"
            if actual_probe_type == "live":
                PROBE_LIVE_CALLS.labels(model_id=model_id, result=result_label).inc()
            else:
                PROBE_SYNTHETIC_CALLS.labels(model_id=model_id, result=result_label).inc()

            # Persist telemetry
            if self._sf is not None:
                await TelemetryWriter.write(
                    self._sf,
                    model_id=model_id,
                    is_healthy=healthy,
                    latency_ms=latency_ms if healthy else None,
                    consecutive_failures=self._consecutive_failures[model_id],
                    probe_type=actual_probe_type,
                )

        return results

    async def _probe(
        self, adapter, model_id: str, preferred_type: str, failure_threshold: int
    ) -> tuple[bool, str]:
        """Run one probe. For synthetic-first types, escalate to live if synthetic fails.

        Returns (is_healthy, actual_probe_type_used).
        """
        if preferred_type == "live":
            try:
                healthy = await adapter.health_check(model_id)
                return healthy, "live"
            except Exception as exc:
                log.warning("health_probe_error", model_id=model_id, probe_type="live", error=str(exc))
                return False, "live"

        # Synthetic-first path
        try:
            await adapter.count_tokens(model_id, _PROBE_MESSAGES)
            # Synthetic succeeded — model API is reachable
            return True, "synthetic"
        except Exception:
            log.debug("synthetic_probe_failed_escalating", model_id=model_id)

        # Escalate to live
        try:
            healthy = await adapter.health_check(model_id)
            return healthy, "live"
        except Exception as exc:
            log.warning("health_probe_error", model_id=model_id, probe_type="live", error=str(exc))
            return False, "live"

    async def _classify_tiers(self) -> tuple[set[str], set[str]]:
        """Return (tier_a_ids, tier_b_ids). Tier C is implicitly everything else.

        Tier A — consecutive_failures > 0 → always probe live.
        Tier B — no recent telemetry (never probed, or not probed in last 30 min).
        Tier C — recently probed and healthy → 10% sample (caller uses else: branch).
        """
        if self._sf is None:
            # No DB — all models default to Tier B (synthetic first)
            all_ids = {s.model_id for s in self._registry.list_enabled()}
            return set(), all_ids

        try:
            from tidus.registry.telemetry_reader import TelemetryReader
            snapshots = await TelemetryReader().get_all_snapshots(self._sf)
        except Exception as exc:
            log.warning("tier_classification_failed", error=str(exc))
            all_ids = {s.model_id for s in self._registry.list_enabled()}
            return set(), all_ids

        now = datetime.now(UTC)
        tier_a: set[str] = set()
        tier_b: set[str] = set()

        for spec in self._registry.list_enabled():
            mid = spec.model_id
            snap = snapshots.get(mid)

            if snap is None:
                tier_b.add(mid)
                continue

            if snap.consecutive_failures > 0:
                tier_a.add(mid)
                continue

            measured = snap.measured_at
            if measured.tzinfo is None:
                measured = measured.replace(tzinfo=UTC)

            if now - measured > _TIER_B_WINDOW:
                tier_b.add(mid)
            # else: Tier C — caller handles via else: branch

        return tier_a, tier_b
