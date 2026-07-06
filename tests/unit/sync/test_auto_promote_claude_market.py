"""Unit tests for dark promotion of Claude-discovered models in AutoPromoter."""

from __future__ import annotations

from datetime import UTC, datetime

import yaml

from tidus.sync.ai_verifier import (
    ClaudeMarketPriceVerifier,
    MarketVerificationResult,
)
from tidus.sync.auto_promote import AutoPromoter
from tidus.sync.discovery.base import DiscoveredModel


class _AcceptAll(ClaudeMarketPriceVerifier):
    def __init__(self):
        pass

    @property
    def is_available(self):
        return True

    async def verify(self, candidates):
        return MarketVerificationResult(accepted=list(candidates))


class _RejectAll(ClaudeMarketPriceVerifier):
    def __init__(self):
        pass

    @property
    def is_available(self):
        return True

    async def verify(self, candidates):
        return MarketVerificationResult(
            rejected=[(c, "price mismatch") for c in candidates]
        )


def _claude_model(mid: str = "acme-ultra-2") -> DiscoveredModel:
    return DiscoveredModel(
        model_id=mid,
        vendor_id=mid,
        vendor="acme",
        display_name=mid,
        source_name="claude-market",
        retrieved_at=datetime.now(UTC),
        raw_metadata={
            "claude_sourced": True,
            "price_in_per_1k": 0.005,
            "price_out_per_1k": 0.02,
            "purpose": "x",
            "positioning": "flagship",
            "sources": ["https://acme.ai"],
        },
    )


async def test_claude_model_promoted_dark(tmp_path):
    promoter = AutoPromoter(
        auto_yaml_path=tmp_path / "models.auto.yaml",
        ai_verifier=None,
        market_verifier=_AcceptAll(),
    )
    await promoter.run(discovered=[_claude_model()], hand_curated_ids=set())

    written = yaml.safe_load((tmp_path / "models.auto.yaml").read_text(encoding="utf-8"))
    spec = next(m for m in written["models"] if m["model_id"] == "acme-ultra-2")
    assert spec["route_source"] == "claude_market"
    assert spec["route_id"] is None
    assert spec["input_price"] == 0.005
    assert spec["output_price"] == 0.02


async def test_claude_model_rejected_by_market_verifier_not_written(tmp_path):
    promoter = AutoPromoter(
        auto_yaml_path=tmp_path / "models.auto.yaml",
        ai_verifier=None,
        market_verifier=_RejectAll(),
    )
    result = await promoter.run(discovered=[_claude_model()], hand_curated_ids=set())

    written = yaml.safe_load((tmp_path / "models.auto.yaml").read_text(encoding="utf-8"))
    assert written["models"] == []
    assert result.ai_rejected == 1


async def test_claude_model_fail_open_without_market_verifier(tmp_path):
    """No market_verifier wired → accept-through (fail-open), same as the
    OpenRouter path's behavior when ai_verifier is absent."""
    promoter = AutoPromoter(
        auto_yaml_path=tmp_path / "models.auto.yaml",
        ai_verifier=None,
        market_verifier=None,
    )
    await promoter.run(discovered=[_claude_model()], hand_curated_ids=set())

    written = yaml.safe_load((tmp_path / "models.auto.yaml").read_text(encoding="utf-8"))
    spec = next(m for m in written["models"] if m["model_id"] == "acme-ultra-2")
    assert spec["route_source"] == "claude_market"
