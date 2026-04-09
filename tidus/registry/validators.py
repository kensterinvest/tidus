"""Three-tier validation for catalog revisions.

Tier 1 — SchemaValidator:
    Validates each spec dict can be parsed as a ModelSpec (Pydantic).
    Required fields, non-negative prices, context window > 0, valid enum values.

Tier 2 — InvariantValidator:
    Cross-field semantic checks that Pydantic cannot express:
    - min_complexity ≤ max_complexity (ordering)
    - Local models must have input_price = output_price = 0.0
    - Non-local models must have input_price > 0 (priced models need a price)

Tier 3 — CanaryProbe:
    Live probe of a random sample of up to 3 models.
    Per model: up to canary_max_attempts retries.
    Revision passes if ≥ canary_pass_rate of sampled models pass.
    Results stored in ModelCatalogRevisionORM.canary_results.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Literal

import structlog

from tidus.models.model_registry import ModelSpec

log = structlog.get_logger(__name__)

_COMPLEXITY_ORDER = {"simple": 0, "moderate": 1, "complex": 2, "critical": 3}


# ── Tier 1: Schema ────────────────────────────────────────────────────────────

class SchemaValidator:
    """Validates that each spec_json can be parsed as a valid ModelSpec."""

    def validate(self, spec_dicts: list[dict]) -> list[str]:
        """Return a list of error strings (empty = all valid)."""
        errors = []
        for i, d in enumerate(spec_dicts):
            model_id = d.get("model_id", f"<entry #{i}>")
            try:
                ModelSpec.model_validate(d)
            except Exception as exc:
                errors.append(f"{model_id}: schema error — {exc}")
        return errors


# ── Tier 2: Invariants ────────────────────────────────────────────────────────

class InvariantValidator:
    """Cross-field semantic invariants that Pydantic cannot express."""

    def validate(self, specs: list[ModelSpec]) -> list[str]:
        """Return a list of error strings (empty = all valid)."""
        errors = []
        for spec in specs:
            errors.extend(self._check(spec))
        return errors

    def _check(self, spec: ModelSpec) -> list[str]:
        errs = []

        # min_complexity must be ≤ max_complexity in the ordering
        min_ord = _COMPLEXITY_ORDER.get(spec.min_complexity)
        max_ord = _COMPLEXITY_ORDER.get(spec.max_complexity)
        if min_ord is not None and max_ord is not None and min_ord > max_ord:
            errs.append(
                f"{spec.model_id}: min_complexity={spec.min_complexity!r} > "
                f"max_complexity={spec.max_complexity!r}"
            )

        # Local models must be free
        if spec.is_local and (spec.input_price != 0.0 or spec.output_price != 0.0):
            errs.append(
                f"{spec.model_id}: is_local=True but input_price={spec.input_price} "
                f"or output_price={spec.output_price} (local models must be free)"
            )

        # Non-local, non-deprecated models should have a price set
        if not spec.is_local and not spec.deprecated and spec.input_price == 0.0:
            errs.append(
                f"{spec.model_id}: non-local model has input_price=0.0 "
                "(set a price or mark is_local=True)"
            )

        return errs


# ── Tier 3: Canary Probe ─────────────────────────────────────────────────────

@dataclass
class CanaryProbeResult:
    """Result of a single canary probe attempt for one model."""

    model_id: str
    attempts: int
    successes: int
    failure_reasons: list[str] = field(default_factory=list)
    verdict: Literal["pass", "fail", "skip"] = "fail"
    # skip = adapter not available or model not in enabled list


class CanaryProbe:
    """Live probe of a random sample of models to gate revision promotion.

    Samples up to `sample_size` models from the enabled set.
    Each model is tried up to `max_attempts` times with `retry_delay_seconds` between.
    A model passes if any attempt succeeds.
    The revision passes if ≥ `pass_rate` fraction of sampled models pass.
    """

    def __init__(
        self,
        sample_size: int = 3,
        max_attempts: int = 3,
        retry_delay_seconds: float = 30.0,
        pass_rate: float = 0.67,
    ) -> None:
        self._sample_size = sample_size
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay_seconds
        self._pass_rate = pass_rate

    async def run(self, specs: list[ModelSpec]) -> tuple[bool, list[CanaryProbeResult]]:
        """Probe a random sample and return (revision_passes, results).

        Returns True if enough models pass or if no models could be sampled.
        """
        try:
            from tidus.adapters.adapter_factory import get_adapter
        except ImportError:
            log.warning("canary_probe_skipped", reason="adapter_factory not available")
            return True, []

        # Sample from enabled, non-deprecated models that have an adapter
        candidates = [s for s in specs if s.enabled and not s.deprecated and not s.is_local]
        sample = random.sample(candidates, min(self._sample_size, len(candidates)))

        if not sample:
            log.info("canary_probe_skipped", reason="no_eligible_models")
            return True, []

        tasks = [self._probe_one(spec, get_adapter) for spec in sample]
        results: list[CanaryProbeResult] = await asyncio.gather(*tasks)

        passes = sum(1 for r in results if r.verdict == "pass")
        skips = sum(1 for r in results if r.verdict == "skip")
        eligible = len(results) - skips

        if eligible == 0:
            log.info("canary_probe_all_skipped")
            return True, results

        rate = passes / eligible
        revision_passes = rate >= self._pass_rate

        log.info(
            "canary_probe_complete",
            passes=passes,
            eligible=eligible,
            pass_rate=round(rate, 2),
            required_rate=self._pass_rate,
            revision_passes=revision_passes,
        )
        return revision_passes, results

    async def _probe_one(self, spec: ModelSpec, get_adapter) -> CanaryProbeResult:
        result = CanaryProbeResult(model_id=spec.model_id, attempts=0, successes=0)
        try:
            adapter = get_adapter(spec.vendor)
        except KeyError:
            result.verdict = "skip"
            return result

        for attempt in range(self._max_attempts):
            result.attempts += 1
            try:
                is_healthy = await adapter.health_check(spec.model_id)
                if is_healthy:
                    result.successes += 1
                    result.verdict = "pass"
                    return result
                else:
                    result.failure_reasons.append(f"attempt {attempt + 1}: health_check=False")
            except Exception as exc:
                result.failure_reasons.append(f"attempt {attempt + 1}: {exc}")
                log.warning(
                    "canary_probe_attempt_failed",
                    model_id=spec.model_id,
                    attempt=attempt + 1,
                    error=str(exc),
                )

            if attempt < self._max_attempts - 1:
                await asyncio.sleep(self._retry_delay)

        result.verdict = "fail"
        return result
