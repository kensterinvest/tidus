"""Pure merge functions for the three-layer EffectiveModelSpec computation.

Merge precedence (priority 0 = highest):

  0  emergency_freeze_revision   → return base catalog unchanged; no overrides, no telemetry
  1  hard_disable_model          → enabled=False; immune to telemetry re-enable
  2  enabled=False in base       → base wins; telemetry cannot re-enable
  3  force_local_only            → marks non-local model as enabled=False for this scope
     force_tier_ceiling          → marks model enabled=False if tier > max_tier
  4  price_multiplier            → scales input_price and output_price
  5  latency_p50_ms (telemetry)  → overrides base if telemetry is "fresh"
  6  base fields                 → everything else (pricing, deprecated, capabilities)

All functions are pure (no DB calls, no side effects) and operate on immutable
ModelSpec via model_copy(). They are tested independently of the registry.
"""

from __future__ import annotations

from tidus.db.registry_orm import ModelOverrideORM
from tidus.models.model_registry import ModelSpec
from tidus.models.registry_models import TelemetrySnapshot


def _model_matches(override: ModelOverrideORM, spec: ModelSpec) -> bool:
    """Return True if this override applies to the given model."""
    if override.model_id is not None and override.model_id != spec.model_id:
        return False
    return True


def merge_spec(
    base: ModelSpec,
    overrides: list[ModelOverrideORM],
    telemetry: TelemetrySnapshot | None,
) -> ModelSpec:
    """Compute the EffectiveModelSpec by applying overrides and telemetry to the base.

    Args:
        base:      The base ModelSpec from the active catalog revision.
        overrides: All currently active ModelOverrideORM rows (pre-filtered to is_active=True).
        telemetry: Most recent health probe snapshot for this model, or None if absent.

    Returns:
        A (possibly new) ModelSpec with merged fields. The base is never mutated.
    """
    active = [o for o in overrides if o.is_active]

    # Priority 0: emergency freeze — return base unchanged
    if any(o.override_type == "emergency_freeze_revision" for o in active):
        return base

    # Priority 1: hard_disable_model — immune to telemetry re-enable
    if any(o.override_type == "hard_disable_model" and _model_matches(o, base) for o in active):
        return base.model_copy(update={"enabled": False})

    # Priority 2: enabled=False in base — telemetry cannot re-enable
    if not base.enabled:
        return base

    updates: dict = {}

    # Priority 3: force_local_only / force_tier_ceiling
    for o in active:
        if not _model_matches(o, base):
            continue
        if o.override_type == "force_local_only" and not base.is_local:
            # Non-local model is disabled for this scope
            updates["enabled"] = False
        if o.override_type == "force_tier_ceiling":
            max_tier = int(o.payload.get("max_tier", 4))
            if base.tier.value > max_tier:
                updates["enabled"] = False

    # Priority 4: price_multiplier (last one with matching scope wins)
    # All four price fields are scaled so cost estimates remain consistent
    # even for models where prompt-cache pricing is significant.
    for o in active:
        if o.override_type == "price_multiplier" and _model_matches(o, base):
            mult = float(o.payload.get("multiplier", 1.0))
            updates["input_price"] = base.input_price * mult
            updates["output_price"] = base.output_price * mult
            updates["cache_read_price"] = base.cache_read_price * mult
            updates["cache_write_price"] = base.cache_write_price * mult

    # Priority 5: latency from fresh telemetry
    if (
        telemetry is not None
        and telemetry.staleness == "fresh"
        and telemetry.latency_p50_ms is not None
    ):
        updates["latency_p50_ms"] = telemetry.latency_p50_ms

    # Priority 6: base fields — untouched (they are the remaining fields of base)

    if not updates:
        return base
    return base.model_copy(update=updates)


def apply_price_multiplier(spec: ModelSpec, multiplier: float) -> ModelSpec:
    """Convenience: return a new ModelSpec with input/output prices scaled by multiplier."""
    return spec.model_copy(update={
        "input_price": spec.input_price * multiplier,
        "output_price": spec.output_price * multiplier,
    })
