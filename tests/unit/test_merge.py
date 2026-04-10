"""Unit tests for the pure merge functions in tidus.registry.merge.

Covers all 6 precedence levels and edge cases:
  - Emergency freeze supersedes everything
  - Hard disable immune to telemetry
  - Disabled base immune to telemetry re-enable
  - force_local_only and force_tier_ceiling
  - price_multiplier scaling
  - Fresh telemetry latency applied; stale/unknown ignored
  - Base fields never mutated (immutability)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tidus.models.model_registry import ModelSpec, ModelTier, TokenizerType
from tidus.models.registry_models import TelemetrySnapshot
from tidus.registry.merge import apply_price_multiplier, merge_spec

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_override(override_type: str, payload: dict | None = None, model_id: str | None = None):
    o = MagicMock()
    o.override_type = override_type
    o.payload = payload or {}
    o.model_id = model_id
    o.is_active = True
    return o


def make_snapshot(staleness: str, latency_p50_ms: int | None = 100) -> TelemetrySnapshot:
    from datetime import UTC, datetime, timedelta
    if staleness == "fresh":
        measured_at = datetime.now(UTC) - timedelta(hours=1)
    elif staleness == "unknown":
        measured_at = datetime.now(UTC) - timedelta(hours=48)
    else:
        measured_at = datetime.now(UTC) - timedelta(hours=96)
    return TelemetrySnapshot(
        model_id="gpt-4o",
        measured_at=measured_at,
        latency_p50_ms=latency_p50_ms,
        is_healthy=True,
        consecutive_failures=0,
        staleness=staleness,
    )


@pytest.fixture
def base_spec():
    return ModelSpec(
        model_id="gpt-4o",
        display_name="GPT-4o",
        vendor="openai",
        provider="openai",
        enabled=True,
        is_local=False,
        tier=ModelTier.mid,
        tokenizer=TokenizerType.tiktoken_o200k,
        max_context=128000,
        input_price=5.0,
        output_price=15.0,
    )


# ── Priority 0: emergency freeze ──────────────────────────────────────────────

def test_emergency_freeze_returns_base_unchanged(base_spec):
    """Emergency freeze must suppress all overrides and telemetry."""
    freeze = make_override("emergency_freeze_revision")
    price_override = make_override("price_multiplier", {"multiplier": 0.5})
    telemetry = make_snapshot("fresh", latency_p50_ms=50)

    result = merge_spec(base_spec, [freeze, price_override], telemetry)

    assert result is base_spec  # exact same object, not a copy
    assert result.input_price == 5.0
    assert result.latency_p50_ms == base_spec.latency_p50_ms


# ── Priority 1: hard_disable_model ───────────────────────────────────────────

def test_hard_disable_sets_enabled_false(base_spec):
    disable = make_override("hard_disable_model", model_id="gpt-4o")
    result = merge_spec(base_spec, [disable], None)
    assert result.enabled is False


def test_hard_disable_immune_to_fresh_telemetry(base_spec):
    """hard_disable must override even 'fresh' telemetry that would re-enable."""
    disable = make_override("hard_disable_model", model_id="gpt-4o")
    telemetry = make_snapshot("fresh")
    result = merge_spec(base_spec, [disable], telemetry)
    assert result.enabled is False


# ── Priority 2: base enabled=False ───────────────────────────────────────────

def test_disabled_base_immune_to_telemetry(base_spec):
    disabled_base = base_spec.model_copy(update={"enabled": False})
    telemetry = make_snapshot("fresh")
    result = merge_spec(disabled_base, [], telemetry)
    assert result is disabled_base
    assert result.enabled is False


# ── Priority 3: force_local_only / force_tier_ceiling ────────────────────────

def test_force_local_only_disables_non_local(base_spec):
    assert not base_spec.is_local
    override = make_override("force_local_only")
    result = merge_spec(base_spec, [override], None)
    assert result.enabled is False


def test_force_local_only_does_not_disable_local_model():
    """force_local_only must not affect models that are already local."""
    local_spec = ModelSpec(
        model_id="ollama-llama3",
        display_name="Llama 3 (local)",
        vendor="ollama",
        provider="ollama",
        enabled=True,
        is_local=True,
        tier=ModelTier.local,
        tokenizer=TokenizerType.ollama,
        max_context=8192,
        input_price=0.0,
        output_price=0.0,
    )
    override = make_override("force_local_only")
    result = merge_spec(local_spec, [override], None)
    assert result.enabled is True


def test_force_tier_ceiling_disables_above_max(base_spec):
    # ModelTier.mid == 2; max_tier=1 means only tier ≤ 1 (premium) is allowed
    assert base_spec.tier == ModelTier.mid  # value = 2
    override = make_override("force_tier_ceiling", {"max_tier": 1})
    result = merge_spec(base_spec, [override], None)
    assert result.enabled is False


def test_force_tier_ceiling_allows_at_or_below_max(base_spec):
    # ModelTier.mid == 2; max_tier=2 means mid is still allowed
    assert base_spec.tier == ModelTier.mid  # value = 2
    override = make_override("force_tier_ceiling", {"max_tier": 2})
    result = merge_spec(base_spec, [override], None)
    assert result.enabled is True


# ── Priority 4: price_multiplier ─────────────────────────────────────────────

def test_price_multiplier_scales_prices(base_spec):
    override = make_override("price_multiplier", {"multiplier": 2.0}, model_id="gpt-4o")
    result = merge_spec(base_spec, [override], None)
    assert result.input_price == pytest.approx(10.0)
    assert result.output_price == pytest.approx(30.0)


def test_price_multiplier_fractional(base_spec):
    override = make_override("price_multiplier", {"multiplier": 0.5}, model_id="gpt-4o")
    result = merge_spec(base_spec, [override], None)
    assert result.input_price == pytest.approx(2.5)
    assert result.output_price == pytest.approx(7.5)


# ── Priority 5: telemetry latency ────────────────────────────────────────────

def test_fresh_telemetry_latency_applied(base_spec):
    telemetry = make_snapshot("fresh", latency_p50_ms=42)
    result = merge_spec(base_spec, [], telemetry)
    assert result.latency_p50_ms == 42


def test_unknown_staleness_telemetry_not_applied(base_spec):
    original_latency = base_spec.latency_p50_ms
    telemetry = make_snapshot("unknown", latency_p50_ms=999)
    result = merge_spec(base_spec, [], telemetry)
    assert result.latency_p50_ms == original_latency


def test_expired_telemetry_not_applied(base_spec):
    original_latency = base_spec.latency_p50_ms
    telemetry = make_snapshot("expired", latency_p50_ms=999)
    result = merge_spec(base_spec, [], telemetry)
    assert result.latency_p50_ms == original_latency


# ── Non-mutation guarantee ────────────────────────────────────────────────────

def test_base_is_never_mutated(base_spec):
    """merge_spec must return a new object, never modify base in-place."""
    original_input_price = base_spec.input_price
    original_enabled = base_spec.enabled

    price_override = make_override("price_multiplier", {"multiplier": 3.0}, model_id="gpt-4o")
    disable = make_override("hard_disable_model", model_id="gpt-4o")

    merge_spec(base_spec, [price_override], None)
    merge_spec(base_spec, [disable], None)

    assert base_spec.input_price == original_input_price
    assert base_spec.enabled == original_enabled


# ── apply_price_multiplier convenience function ───────────────────────────────

def test_apply_price_multiplier_convenience(base_spec):
    result = apply_price_multiplier(base_spec, 1.5)
    assert result.input_price == pytest.approx(7.5)
    assert result.output_price == pytest.approx(22.5)
    assert base_spec.input_price == 5.0  # base not mutated


# ── Price multiplier: cache pricing fields ────────────────────────────────────

def test_price_multiplier_scales_all_four_prices():
    """price_multiplier must scale input, output, cache_read, AND cache_write prices."""
    spec = ModelSpec(
        model_id="cache-model",
        display_name="Cache Model",
        vendor="openai",
        provider="openai",
        enabled=True,
        is_local=False,
        tier=ModelTier.mid,
        tokenizer=TokenizerType.tiktoken_o200k,
        max_context=128000,
        input_price=10.0,
        output_price=30.0,
        cache_read_price=2.0,
        cache_write_price=5.0,
    )
    override = make_override("price_multiplier", {"multiplier": 2.0}, model_id="cache-model")
    result = merge_spec(spec, [override], None)

    assert result.input_price == pytest.approx(20.0)
    assert result.output_price == pytest.approx(60.0)
    assert result.cache_read_price == pytest.approx(4.0)
    assert result.cache_write_price == pytest.approx(10.0)


def test_price_multiplier_does_not_affect_zero_cache_prices():
    """Models with no cache pricing (default 0.0) stay at 0 after multiplier."""
    spec = ModelSpec(
        model_id="nocache-model",
        display_name="No Cache",
        vendor="openai",
        provider="openai",
        enabled=True,
        is_local=False,
        tier=ModelTier.mid,
        tokenizer=TokenizerType.tiktoken_o200k,
        max_context=128000,
        input_price=5.0,
        output_price=15.0,
        # cache_read_price and cache_write_price default to 0.0
    )
    override = make_override("price_multiplier", {"multiplier": 3.0}, model_id="nocache-model")
    result = merge_spec(spec, [override], None)

    assert result.cache_read_price == pytest.approx(0.0)
    assert result.cache_write_price == pytest.approx(0.0)
