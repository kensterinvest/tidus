"""Unit tests for AutoPromoter and ModelRegistry's auto.yaml merge."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from tidus.router.registry import ModelRegistry
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
    def test_priced_known_vendor_is_promoted(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = promoter.run(
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

    def test_hand_curated_model_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = promoter.run(
            discovered=[_disc(model_id="gemini-2.5-pro")],
            hand_curated_ids={"gemini-2.5-pro"},
        )
        assert result.promoted == []
        assert result.skipped_known == 1

    def test_unknown_vendor_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = promoter.run(
            discovered=[_disc(model_id="exotic-model", vendor="some-new-startup")],
            hand_curated_ids=set(),
        )
        assert result.promoted == []
        assert result.skipped_unknown_vendor == 1

    def test_zero_price_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = promoter.run(
            discovered=[_disc(model_id="freebie", prompt="0", completion="0")],
            hand_curated_ids=set(),
        )
        assert result.promoted == []
        assert result.skipped_no_price == 1

    def test_missing_price_field_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = promoter.run(
            discovered=[_disc(model_id="incomplete", prompt=None, completion="0.001")],
            hand_curated_ids=set(),
        )
        assert result.promoted == []
        assert result.skipped_no_price == 1

    def test_preview_variant_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = promoter.run(
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

    def test_free_variant_is_skipped(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        result = promoter.run(
            discovered=[_disc(model_id="gemini-2.5-pro",
                              vendor_id="google/gemini-2.5-pro:free")],
            hand_curated_ids=set(),
        )
        assert result.promoted == []
        assert result.skipped_variant == 1


# ── File output ───────────────────────────────────────────────────────────────

class TestFileOutput:
    def test_writes_well_formed_yaml(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        promoter.run(
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

    def test_empty_promotion_still_writes_file_with_header(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        promoter.run(discovered=[], hand_curated_ids=set())
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "DO NOT EDIT BY HAND" in text
        data = yaml.safe_load(text)
        assert data["models"] == []

    def test_rewrites_clean_each_run(self, tmp_path):
        out = tmp_path / "models.auto.yaml"
        promoter = AutoPromoter(auto_yaml_path=out)
        promoter.run(
            discovered=[_disc(model_id="gemini-3.5-pro")],
            hand_curated_ids=set(),
        )
        # Second run, gemini gone, claude shows up
        promoter.run(
            discovered=[_disc(model_id="claude-opus-5", vendor="anthropic")],
            hand_curated_ids=set(),
        )
        data = yaml.safe_load(out.read_text(encoding="utf-8"))
        ids = {m["model_id"] for m in data["models"]}
        assert ids == {"claude-opus-5"}


# ── Kill-switch ───────────────────────────────────────────────────────────────

def test_disabled_promoter_is_noop(tmp_path):
    out = tmp_path / "models.auto.yaml"
    promoter = AutoPromoter(auto_yaml_path=out, enabled=False)
    result = promoter.run(
        discovered=[_disc(model_id="gemini-3.5-pro")],
        hand_curated_ids=set(),
    )
    assert result.promoted == []
    assert not out.exists()


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
