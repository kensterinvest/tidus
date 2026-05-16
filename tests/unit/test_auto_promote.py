"""Unit tests for AutoPromoter and ModelRegistry's auto.yaml merge."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from tidus.router.registry import ModelRegistry
from tidus.sync.ai_verifier import (
    ClaudeDiscoveryVerifier,
    DiscoveryVerificationResult,
    RejectedCandidate,
)
from tidus.sync.auto_promote import AutoPromoter
from tidus.sync.discovery.base import DiscoveredModel


def _disc(
    *,
    model_id: str,
    vendor: str = "google",
    prompt: str | None = "0.000001",
    completion: str | None = "0.000005",
    context_length: int | None = 100000,
    vendor_id: str | None = None,
) -> DiscoveredModel:
    pricing: dict = {}
    if prompt is not None:
        pricing["prompt"] = prompt
    if completion is not None:
        pricing["completion"] = completion
    return DiscoveredModel(
        model_id=model_id,
        vendor_id=vendor_id or f"{vendor}/{model_id}",
        vendor=vendor,
        display_name=f"{vendor.title()}: {model_id}",
        source_name="openrouter-discovery",
        retrieved_at=datetime.now(UTC),
        raw_metadata={"pricing": pricing, "context_length": context_length},
    )


# ── Promotion rules ───────────────────────────────────────────────────────────

class TestPromotionFilters:
    async def test_priced_known_vendor_is_promoted(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = await promoter.run(
            discovered=[_disc(model_id="gemini-3.5-pro")],
            hand_curated_ids=set(),
        )
        assert len(result.promoted) == 1
        spec = result.promoted[0]
        assert spec.model_id == "gemini-3.5-pro"
        assert spec.vendor == "google"
        # Per-token 0.000001 → per-1K 0.001
        assert spec.input_price == pytest.approx(0.001)
        assert spec.output_price == pytest.approx(0.005)
        # Conservative defaults
        assert int(spec.tier) == 3
        assert [c.value for c in spec.capabilities] == ["chat"]
        assert spec.max_complexity == "moderate"
        assert spec.fallbacks == []
        assert spec.enabled is True
        assert spec.deprecated is False

    async def test_hand_curated_model_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = await promoter.run(
            discovered=[_disc(model_id="gemini-2.5-pro")],
            hand_curated_ids={"gemini-2.5-pro"},
        )
        assert result.promoted == []
        assert result.skipped_known == 1

    async def test_unknown_vendor_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = await promoter.run(
            discovered=[_disc(model_id="exotic-model", vendor="some-new-startup")],
            hand_curated_ids=set(),
        )
        assert result.promoted == []
        assert result.skipped_unknown_vendor == 1

    async def test_zero_price_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = await promoter.run(
            discovered=[_disc(model_id="freebie", prompt="0", completion="0")],
            hand_curated_ids=set(),
        )
        assert result.promoted == []
        assert result.skipped_no_price == 1

    async def test_missing_price_field_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = await promoter.run(
            discovered=[_disc(model_id="incomplete", prompt=None, completion="0.001")],
            hand_curated_ids=set(),
        )
        assert result.promoted == []
        assert result.skipped_no_price == 1

    async def test_preview_variant_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = await promoter.run(
            discovered=[
                _disc(model_id="gemini-x-preview", vendor_id="google/gemini-x-preview"),
                _disc(model_id="claude-test", vendor="anthropic",
                      vendor_id="anthropic/claude-test"),
                _disc(model_id="kimi-nightly", vendor="moonshot",
                      vendor_id="moonshotai/kimi-nightly"),
            ],
            hand_curated_ids=set(),
        )
        assert result.promoted == []
        assert result.skipped_variant == 3

    async def test_free_variant_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = await promoter.run(
            discovered=[_disc(model_id="gemini-2.5-pro",
                              vendor_id="google/gemini-2.5-pro:free")],
            hand_curated_ids=set(),
        )
        assert result.promoted == []
        assert result.skipped_variant == 1


# ── File output ───────────────────────────────────────────────────────────────

class TestFileOutput:
    async def test_writes_well_formed_yaml(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        await promoter.run(
            discovered=[
                _disc(model_id="gemini-3.5-pro"),
                _disc(model_id="claude-opus-5", vendor="anthropic"),
            ],
            hand_curated_ids=set(),
        )

        text = out.read_text(encoding="utf-8")
        assert "DO NOT EDIT BY HAND" in text
        data = yaml.safe_load(text)
        ids = {m["model_id"] for m in data["models"]}
        assert ids == {"gemini-3.5-pro", "claude-opus-5"}

    async def test_empty_promotion_still_writes_file_with_header(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        await promoter.run(discovered=[], hand_curated_ids=set())
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "DO NOT EDIT BY HAND" in text
        data = yaml.safe_load(text)
        assert data["models"] == []

    async def test_rewrites_clean_each_run(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        await promoter.run(
            discovered=[_disc(model_id="gemini-3.5-pro")],
            hand_curated_ids=set(),
        )
        # Second run, gemini gone, claude shows up
        await promoter.run(
            discovered=[_disc(model_id="claude-opus-5", vendor="anthropic")],
            hand_curated_ids=set(),
        )
        data = yaml.safe_load(out.read_text(encoding="utf-8"))
        ids = {m["model_id"] for m in data["models"]}
        assert ids == {"claude-opus-5"}


# ── Kill-switch ───────────────────────────────────────────────────────────────

async def test_disabled_promoter_is_noop(tmp_path):
    out = tmp_path / "models.auto.yaml"
    promoter = AutoPromoter(auto_yaml_path=out, enabled=False)
    result = await promoter.run(
        discovered=[_disc(model_id="gemini-3.5-pro")],
        hand_curated_ids=set(),
    )
    assert result.promoted == []
    assert not out.exists()


# ── AI verifier integration ───────────────────────────────────────────────────

def _fake_verifier(verdict_result: DiscoveryVerificationResult) -> ClaudeDiscoveryVerifier:
    """Return a ClaudeDiscoveryVerifier whose .verify() is mocked to return verdict_result."""
    v = ClaudeDiscoveryVerifier(api_key="sk-fake")
    v.verify = AsyncMock(return_value=verdict_result)  # type: ignore[method-assign]
    return v


class TestAIVerifierIntegration:
    async def test_no_verifier_passes_through(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out, ai_verifier=None)
        result = await promoter.run(
            discovered=[_disc(model_id="gemini-3.5-pro")],
            hand_curated_ids=set(),
        )
        assert len(result.promoted) == 1
        assert result.ai_rejected == 0

    async def test_verifier_accepts_all(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        verifier = _fake_verifier(DiscoveryVerificationResult())  # empty rejected

        # Pre-populate accepted with the spec the promoter is about to build,
        # so the merge sees a non-empty accepted list (but the merge logic
        # only consults rejected to drop entries — empty rejected = accept all).
        promoter = AutoPromoter(auto_yaml_path=out, ai_verifier=verifier)
        result = await promoter.run(
            discovered=[_disc(model_id="gemini-3.5-pro")],
            hand_curated_ids=set(),
        )

        assert len(result.promoted) == 1
        assert result.ai_rejected == 0
        verifier.verify.assert_awaited_once()

    async def test_verifier_rejects_implausible_model(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        # Promoter will build a spec for "fake-flagship-claude" — rule-based
        # filters can't catch it (vendor=anthropic is in the allow-list,
        # price is non-zero, no skip-pattern match). The verifier rejects it.
        from tidus.sync.ai_verifier import DiscoveryCandidate
        rejected = DiscoveryVerificationResult(
            rejected=[
                RejectedCandidate(
                    candidate=DiscoveryCandidate(
                        model_id="fake-flagship-claude",
                        vendor="anthropic",
                        openrouter_id="anthropic/fake-flagship-claude",
                        display_name="Fake",
                        input_price_per_1m=1.0,
                        output_price_per_1m=5.0,
                    ),
                    reasoning="No public announcement of this model from Anthropic.",
                )
            ]
        )
        verifier = _fake_verifier(rejected)

        promoter = AutoPromoter(auto_yaml_path=out, ai_verifier=verifier)
        result = await promoter.run(
            discovered=[_disc(model_id="fake-flagship-claude", vendor="anthropic")],
            hand_curated_ids=set(),
        )

        assert result.promoted == []
        assert result.ai_rejected == 1

        # File still written, but empty
        data = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert data["models"] == []

    async def test_verifier_unavailable_passes_through(self, tmp_path):
        # is_available=False → integration treats it as "no verifier"
        verifier = ClaudeDiscoveryVerifier(api_key="", enabled=True)  # no key → unavailable
        assert verifier.is_available is False

        promoter = AutoPromoter(
            auto_yaml_path=tmp_path / "models.auto.yaml",
            ai_verifier=verifier,
        )
        result = await promoter.run(
            discovered=[_disc(model_id="gemini-3.5-pro")],
            hand_curated_ids=set(),
        )
        assert len(result.promoted) == 1
        assert result.ai_rejected == 0

    async def test_total_evaluated_includes_ai_rejected(self, tmp_path):
        from tidus.sync.ai_verifier import DiscoveryCandidate
        rejected = DiscoveryVerificationResult(
            rejected=[
                RejectedCandidate(
                    candidate=DiscoveryCandidate(
                        model_id="m1",
                        vendor="google",
                        openrouter_id="google/m1",
                        display_name=None,
                        input_price_per_1m=1.0,
                        output_price_per_1m=2.0,
                    ),
                    reasoning="rejected",
                )
            ]
        )
        verifier = _fake_verifier(rejected)
        promoter = AutoPromoter(auto_yaml_path=tmp_path / "auto.yaml", ai_verifier=verifier)
        result = await promoter.run(
            discovered=[
                _disc(model_id="m1"),  # rule-pass, AI-reject
                _disc(model_id="m2", vendor="some-new-startup"),  # vendor-skip
            ],
            hand_curated_ids=set(),
        )
        # 1 ai_rejected + 1 unknown_vendor = 2 total_evaluated
        assert result.ai_rejected == 1
        assert result.skipped_unknown_vendor == 1
        assert result.total_evaluated == 2


# ── ModelRegistry merge ───────────────────────────────────────────────────────

def _write_yaml(path: Path, models: list[dict]) -> None:
    path.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")


def _minimal_model(model_id: str, vendor: str = "google", input_price: float = 0.001) -> dict:
    return {
        "model_id": model_id,
        "display_name": model_id,
        "vendor": vendor,
        "tier": 3,
        "max_context": 100000,
        "input_price": input_price,
        "output_price": input_price * 3,
        "tokenizer": "google",
        "capabilities": ["chat"],
        "min_complexity": "simple",
        "max_complexity": "moderate",
    }


class TestModelRegistryMerge:
    def test_merges_auto_yaml_when_present(self, tmp_path):
        primary = tmp_path / "models.yaml"
        auto = tmp_path / "models.auto.yaml"
        _write_yaml(primary, [_minimal_model("vetted-model")])
        _write_yaml(auto, [_minimal_model("auto-model")])

        registry = ModelRegistry.load(primary, auto_path=auto)
        ids = {s.model_id for s in registry.list_all()}
        assert ids == {"vetted-model", "auto-model"}

    def test_hand_curated_wins_on_conflict(self, tmp_path):
        primary = tmp_path / "models.yaml"
        auto = tmp_path / "models.auto.yaml"
        _write_yaml(primary, [_minimal_model("gemini-2.5-pro", input_price=0.00125)])
        _write_yaml(auto, [_minimal_model("gemini-2.5-pro", input_price=999.0)])

        registry = ModelRegistry.load(primary, auto_path=auto)
        spec = registry.get("gemini-2.5-pro")
        assert spec is not None
        # Hand-curated price (0.00125), NOT the auto entry's bogus 999.0.
        assert spec.input_price == pytest.approx(0.00125)

    def test_missing_auto_file_is_silently_ignored(self, tmp_path):
        primary = tmp_path / "models.yaml"
        _write_yaml(primary, [_minimal_model("only-vetted")])
        # auto_path points to a path that doesn't exist
        registry = ModelRegistry.load(primary, auto_path=tmp_path / "nonexistent.yaml")
        ids = {s.model_id for s in registry.list_all()}
        assert ids == {"only-vetted"}

    def test_auto_path_none_disables_merge(self, tmp_path):
        primary = tmp_path / "models.yaml"
        auto = tmp_path / "models.auto.yaml"
        _write_yaml(primary, [_minimal_model("vetted")])
        _write_yaml(auto, [_minimal_model("auto")])
        registry = ModelRegistry.load(primary, auto_path=None)
        ids = {s.model_id for s in registry.list_all()}
        assert ids == {"vetted"}
