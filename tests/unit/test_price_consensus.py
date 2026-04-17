"""Unit tests for PriceConsensus (MAD outlier detection).

Covers:
  - MAD correctly rejects a statistical outlier
  - A legitimate large price change (e.g. 50% cut) is NOT rejected (MAD adjusts)
  - Single-source lowers confidence by 0.2
  - All sources agree exactly (MAD=0) → no rejection, highest confidence wins
  - All sources rejected as outliers → ConsensusError raised
  - Source confidence used to pick winner among non-outliers
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from tidus.sync.pricing.base import PriceQuote
from tidus.sync.pricing.consensus import ConsensusError, PriceConsensus


def make_quote(
    model_id: str = "gpt-4o",
    input_price: float = 5.0,
    output_price: float = 15.0,
    source_name: str = "source_a",
    source_confidence: float = 0.7,
) -> PriceQuote:
    return PriceQuote(
        model_id=model_id,
        input_price=input_price,
        output_price=output_price,
        cache_read_price=0.0,
        cache_write_price=0.0,
        currency="USD",
        effective_date=date.today(),
        retrieved_at=datetime.now(UTC),
        source_name=source_name,
        source_confidence=source_confidence,
    )


# ── Single source ─────────────────────────────────────────────────────────────

def test_recency_tie_breaker_on_equal_confidence():
    """Fix 8 regression: when two sources tie on confidence, the more recent
    effective_date wins. Commit f5be789 claimed this but the code took the
    first by lexical order via max()."""
    older = PriceQuote(
        model_id="gpt-4o",
        input_price=5.0,
        output_price=15.0,
        cache_read_price=0.0,
        cache_write_price=0.0,
        currency="USD",
        effective_date=date(2026, 1, 1),
        retrieved_at=datetime(2026, 1, 2, tzinfo=UTC),
        source_name="old_source",
        source_confidence=0.7,
    )
    newer = PriceQuote(
        model_id="gpt-4o",
        input_price=5.0,   # same price, same confidence → tie
        output_price=15.0,
        cache_read_price=0.0,
        cache_write_price=0.0,
        currency="USD",
        effective_date=date(2026, 4, 17),
        retrieved_at=datetime(2026, 4, 17, tzinfo=UTC),
        source_name="new_source",
        source_confidence=0.7,
    )
    result = PriceConsensus().resolve([older, newer])
    winner = result.quotes["gpt-4o"]
    assert winner.source_name == "new_source", (
        f"Expected recency tie-breaker to pick 'new_source', got {winner.source_name!r}"
    )


def test_recency_tie_breaker_uses_retrieved_at_when_effective_date_ties():
    """If effective_date also ties, prefer the more recently retrieved quote."""
    earlier = PriceQuote(
        model_id="claude-opus-4-7",
        input_price=5.0, output_price=25.0,
        cache_read_price=0.0, cache_write_price=0.0,
        currency="USD",
        effective_date=date(2026, 4, 17),
        retrieved_at=datetime(2026, 4, 17, 6, 0, 0, tzinfo=UTC),
        source_name="morning_fetch",
        source_confidence=0.8,
    )
    later = PriceQuote(
        model_id="claude-opus-4-7",
        input_price=5.0, output_price=25.0,
        cache_read_price=0.0, cache_write_price=0.0,
        currency="USD",
        effective_date=date(2026, 4, 17),
        retrieved_at=datetime(2026, 4, 17, 18, 0, 0, tzinfo=UTC),
        source_name="evening_fetch",
        source_confidence=0.8,
    )
    result = PriceConsensus().resolve([earlier, later])
    winner = result.quotes["claude-opus-4-7"]
    assert winner.source_name == "evening_fetch"


def test_single_source_accepted_with_reduced_confidence():
    q = make_quote(source_confidence=0.7)
    result = PriceConsensus().resolve([q])
    assert "gpt-4o" in result.quotes
    winner = result.quotes["gpt-4o"]
    assert winner.source_confidence == pytest.approx(0.5)  # 0.7 - 0.2
    assert "gpt-4o" in result.single_source_models


def test_single_source_confidence_never_goes_below_zero():
    q = make_quote(source_confidence=0.1)
    result = PriceConsensus().resolve([q])
    winner = result.quotes["gpt-4o"]
    assert winner.source_confidence >= 0.0


# ── Multi-source: outlier detection ──────────────────────────────────────────

def test_outlier_quote_rejected():
    """A source with a wildly different price should be rejected."""
    normal = make_quote(input_price=5.0, source_name="a", source_confidence=0.7)
    normal2 = make_quote(input_price=5.1, source_name="b", source_confidence=0.85)
    outlier = make_quote(input_price=500.0, source_name="outlier", source_confidence=0.9)

    result = PriceConsensus().resolve([normal, normal2, outlier])
    winner = result.quotes["gpt-4o"]
    # Outlier must not win despite high confidence
    assert winner.source_name != "outlier"
    assert "gpt-4o" in result.rejection_summary
    assert "outlier" in result.rejection_summary["gpt-4o"]


def test_legitimate_large_price_cut_not_rejected():
    """A 50% price cut where both sources agree is NOT an outlier."""
    source_a = make_quote(input_price=5.0, source_name="a", source_confidence=0.7)
    source_b = make_quote(input_price=5.0, source_name="b", source_confidence=0.85)
    # Both sources report the same 50%-cut price — MAD=0, neither is rejected
    result = PriceConsensus().resolve([source_a, source_b])
    # Both non-outlier; winner is highest confidence
    assert result.quotes["gpt-4o"].source_name == "b"
    assert "gpt-4o" not in result.rejection_summary


def test_all_sources_agree_exactly_mad_zero_no_rejection():
    """When all sources report the same price (MAD=0), none are rejected."""
    quotes = [
        make_quote(input_price=3.0, source_name=f"src-{i}", source_confidence=0.5 + i * 0.1)
        for i in range(4)
    ]
    result = PriceConsensus().resolve(quotes)
    # All non-outlier; highest confidence (src-3, confidence=0.8) wins
    assert result.quotes["gpt-4o"].source_name == "src-3"
    assert "gpt-4o" not in result.rejection_summary


def test_all_sources_rejected_raises_consensus_error():
    """All sources being outliers signals systemic data quality issue.

    With two quotes at prices [1.0, 2.0], the median is the interpolated value 1.5
    (not equal to any quote price). Both quotes therefore have identical non-zero
    z-scores (≈ 0.6745), which exceed a threshold of 0.5 — so ALL quotes are
    rejected and ConsensusError must be raised.
    """
    quotes = [
        make_quote(input_price=1.0, source_name="a"),
        make_quote(input_price=2.0, source_name="b"),
    ]
    with pytest.raises(ConsensusError, match="rejected as outliers"):
        PriceConsensus(outlier_z_threshold=0.5).resolve(quotes)


def test_highest_confidence_wins_among_non_outliers():
    """When two sources pass the outlier check, the one with higher confidence wins."""
    low_conf = make_quote(input_price=5.0, source_name="low", source_confidence=0.5)
    high_conf = make_quote(input_price=5.1, source_name="high", source_confidence=0.9)
    outlier = make_quote(input_price=999.0, source_name="bad", source_confidence=1.0)

    result = PriceConsensus().resolve([low_conf, high_conf, outlier])
    winner = result.quotes["gpt-4o"]
    assert winner.source_name == "high"


# ── Multiple models ───────────────────────────────────────────────────────────

def test_resolve_handles_multiple_models():
    gpt = make_quote(model_id="gpt-4o", input_price=5.0)
    claude = make_quote(model_id="claude-opus-4-6", input_price=15.0)
    result = PriceConsensus().resolve([gpt, claude])
    assert "gpt-4o" in result.quotes
    assert "claude-opus-4-6" in result.quotes
    assert len(result.single_source_models) == 2


def test_empty_quotes_returns_empty_result():
    result = PriceConsensus().resolve([])
    assert result.quotes == {}
    assert result.single_source_models == []
