"""Unit tests for three-tier revision validators.

Covers:
  - SchemaValidator: valid dict passes; missing required field fails; negative price fails
  - InvariantValidator: clean spec passes; complexity inversion fails; local + price fails;
    non-local zero-price fails; deprecated model exempt from price check
  - CanaryProbe: skips gracefully when adapter_factory not importable (no live calls)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tidus.models.model_registry import ModelSpec
from tidus.registry.validators import (
    CanaryProbe,
    CanaryProbeResult,
    InvariantValidator,
    SchemaValidator,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_spec_dict(
    model_id: str = "gpt-4o",
    input_price: float = 5.0,
    is_local: bool = False,
    min_complexity: str = "simple",
    max_complexity: str = "critical",
) -> dict:
    return {
        "model_id": model_id,
        "vendor": "openai",
        "tier": 2,
        "max_context": 128_000,
        "input_price": input_price,
        "output_price": 10.0,
        "tokenizer": "tiktoken_o200k",
        "is_local": is_local,
        "enabled": True,
        "deprecated": False,
        "min_complexity": min_complexity,
        "max_complexity": max_complexity,
    }


def _make_spec(**kwargs) -> ModelSpec:
    return ModelSpec.model_validate(_valid_spec_dict(**kwargs))


# ── Tier 1: SchemaValidator ───────────────────────────────────────────────────

class TestSchemaValidator:
    def test_valid_spec_passes(self):
        errors = SchemaValidator().validate([_valid_spec_dict()])
        assert errors == []

    def test_multiple_valid_specs_all_pass(self):
        specs = [_valid_spec_dict("gpt-4o"), _valid_spec_dict("claude-opus-4-6")]
        errors = SchemaValidator().validate(specs)
        assert errors == []

    def test_missing_required_field_fails(self):
        bad = _valid_spec_dict()
        del bad["vendor"]  # vendor is required
        errors = SchemaValidator().validate([bad])
        assert len(errors) == 1
        assert "gpt-4o" in errors[0]

    def test_negative_price_fails(self):
        bad = _valid_spec_dict()
        bad["input_price"] = -1.0
        errors = SchemaValidator().validate([bad])
        assert len(errors) == 1

    def test_zero_context_window_fails(self):
        bad = _valid_spec_dict()
        bad["max_context"] = 0  # must be > 0
        errors = SchemaValidator().validate([bad])
        assert len(errors) == 1

    def test_empty_list_returns_no_errors(self):
        assert SchemaValidator().validate([]) == []

    def test_mixed_valid_and_invalid_reports_only_invalid(self):
        good = _valid_spec_dict("good-model")
        bad = _valid_spec_dict("bad-model")
        bad["input_price"] = -5.0
        errors = SchemaValidator().validate([good, bad])
        assert len(errors) == 1
        assert "bad-model" in errors[0]


# ── Tier 2: InvariantValidator ────────────────────────────────────────────────

class TestInvariantValidator:
    def test_valid_spec_passes(self):
        spec = _make_spec()
        errors = InvariantValidator().validate([spec])
        assert errors == []

    def test_complexity_inversion_fails(self):
        """min_complexity='critical' > max_complexity='simple' should fail."""
        spec = _make_spec(min_complexity="critical", max_complexity="simple")
        errors = InvariantValidator().validate([spec])
        assert len(errors) == 1
        assert "min_complexity" in errors[0]

    def test_valid_complexity_range_passes(self):
        """min='simple', max='moderate' is a valid range."""
        spec = _make_spec(min_complexity="simple", max_complexity="moderate")
        errors = InvariantValidator().validate([spec])
        assert errors == []

    def test_local_model_with_nonzero_price_fails(self):
        spec = ModelSpec.model_validate({
            **_valid_spec_dict(is_local=True),
            "input_price": 1.0,  # local must be free
        })
        errors = InvariantValidator().validate([spec])
        assert any("is_local" in e for e in errors)

    def test_local_model_free_passes(self):
        spec = ModelSpec.model_validate({
            **_valid_spec_dict(is_local=True),
            "input_price": 0.0,
            "output_price": 0.0,
        })
        errors = InvariantValidator().validate([spec])
        assert errors == []

    def test_nonlocal_zero_price_fails(self):
        spec = _make_spec(input_price=0.0)  # non-local + price=0 is invalid
        errors = InvariantValidator().validate([spec])
        assert any("input_price" in e for e in errors)

    def test_deprecated_nonlocal_zero_price_is_exempt(self):
        """Deprecated models are EOL and may have no current price."""
        spec = ModelSpec.model_validate({
            **_valid_spec_dict(input_price=0.0),
            "deprecated": True,
        })
        errors = InvariantValidator().validate([spec])
        assert errors == []

    def test_multiple_specs_collects_all_errors(self):
        bad1 = _make_spec(min_complexity="critical", max_complexity="simple")
        bad2 = ModelSpec.model_validate({
            **_valid_spec_dict("model-b", is_local=True),
            "input_price": 5.0,
        })
        errors = InvariantValidator().validate([bad1, bad2])
        assert len(errors) == 2


# ── Tier 3: CanaryProbe ───────────────────────────────────────────────────────

class TestCanaryProbe:
    @pytest.mark.asyncio
    async def test_skips_gracefully_when_adapter_factory_unavailable(self):
        """When adapter_factory can't be imported, probe returns True (allow revision)."""
        spec = _make_spec()

        with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": None}):
            # ImportError path → returns (True, [])
            probe = CanaryProbe()
            passes, results = await probe.run([spec])

        assert passes is True
        assert results == []

    @pytest.mark.asyncio
    async def test_passes_when_all_samples_healthy(self):
        """When all sampled adapters return healthy, revision passes."""
        spec = _make_spec()

        mock_adapter = AsyncMock()
        mock_adapter.health_check = AsyncMock(return_value=True)

        with patch("tidus.registry.validators.CanaryProbe._probe_one", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = CanaryProbeResult(
                model_id="gpt-4o",
                attempts=1,
                successes=1,
                verdict="pass",
            )
            probe = CanaryProbe(sample_size=1, max_attempts=1, retry_delay_seconds=0)
            passes, results = await probe.run([spec])

        assert passes is True
        assert results[0].verdict == "pass"

    @pytest.mark.asyncio
    async def test_fails_when_too_few_samples_pass(self):
        """When fewer than pass_rate fraction succeed, revision fails.

        Injects a fake adapter_factory into sys.modules so the ImportError guard
        inside CanaryProbe.run() doesn't short-circuit the probe. Then patches
        _probe_one to always return a 'fail' verdict.
        """
        import sys
        from unittest.mock import MagicMock

        specs = [_make_spec(model_id=f"model-{i}") for i in range(3)]

        fake_module = MagicMock()
        fake_module.get_adapter = MagicMock(return_value=AsyncMock())

        with patch.dict(sys.modules, {"tidus.adapters.adapter_factory": fake_module}):
            with patch("tidus.registry.validators.CanaryProbe._probe_one", new_callable=AsyncMock) as mock_probe:
                mock_probe.return_value = CanaryProbeResult(
                    model_id="gpt-4o",
                    attempts=1,
                    successes=0,
                    verdict="fail",
                )
                probe = CanaryProbe(sample_size=3, max_attempts=1, retry_delay_seconds=0, pass_rate=0.67)
                passes, results = await probe.run(specs)

        assert passes is False

    @pytest.mark.asyncio
    async def test_skips_when_no_eligible_models(self):
        """If all specs are local/disabled/deprecated, probe returns True (no sample possible)."""
        local_spec = ModelSpec.model_validate({
            **_valid_spec_dict(is_local=True),
            "input_price": 0.0,
            "output_price": 0.0,
        })
        probe = CanaryProbe()

        with patch("tidus.registry.validators.CanaryProbe._probe_one", new_callable=AsyncMock):
            # Import guard: fake adapter_factory available
            with patch("tidus.registry.validators.CanaryProbe.run", wraps=probe.run):
                # The method filters locals out and returns True when sample is empty
                with patch("tidus.adapters.adapter_factory", create=True):
                    try:
                        from tidus.adapters import adapter_factory  # noqa: F401
                        passes, results = await probe.run([local_spec])
                        # Local models are filtered from candidates
                        assert passes is True
                    except ImportError:
                        # adapter_factory not present in test env — also valid (returns True)
                        passes, results = await probe.run([local_spec])
                        assert passes is True
