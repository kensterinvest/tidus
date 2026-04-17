"""Unit tests for the magazine / pricing-report ranking tie-breaker.

Regression for the bug where Opus 4.6 was ranked #1 and Opus 4.7 was #3 even
though they share the same $15/M blended cost — the original sort had no
tie-breaker so order depended on dict iteration order.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from tidus.reporting.pricing_report import _rank_key


def _spec(mid: str, input_price: float, output_price: float, released: date | None):
    """Minimal duck-typed stand-in for ModelSpec with just the fields _rank_key reads."""
    return SimpleNamespace(
        model_id=mid,
        input_price=input_price,
        output_price=output_price,
        released_at=released,
    )


class TestRankKey:
    def test_higher_blended_cost_ranks_first(self):
        hi = _spec("hi", 0.005, 0.025, date(2025, 1, 1))   # $15/M
        lo = _spec("lo", 0.003, 0.015, date(2026, 1, 1))   # $9/M, newer
        ranked = sorted([lo, hi], key=_rank_key, reverse=True)
        assert [s.model_id for s in ranked] == ["hi", "lo"], (
            "Blended cost must dominate released_at in the ranking"
        )

    def test_tie_broken_by_newer_released_at(self):
        """Opus 4.7 must beat Opus 4.6 at identical $15/M."""
        new = _spec("claude-opus-4-7", 0.005, 0.025, date(2026, 4, 17))
        old = _spec("claude-opus-4-6", 0.005, 0.025, date(2025, 7, 22))
        ranked = sorted([old, new], key=_rank_key, reverse=True)
        assert ranked[0].model_id == "claude-opus-4-7"
        assert ranked[1].model_id == "claude-opus-4-6"

    def test_three_way_tie_at_top_orders_by_release_then_model_id(self):
        """The three $15/M tier-1 models: Opus 4.7 > Opus 4.6 > grok-3-fast."""
        opus_47 = _spec("claude-opus-4-7", 0.005, 0.025, date(2026, 4, 17))
        opus_46 = _spec("claude-opus-4-6", 0.005, 0.025, date(2025, 7, 22))
        grok_fast = _spec("grok-3-fast", 0.005, 0.025, date(2025, 2, 18))
        # Intentionally pass in "wrong" order to make the sort do the work
        ranked = sorted([grok_fast, opus_46, opus_47], key=_rank_key, reverse=True)
        assert [s.model_id for s in ranked] == [
            "claude-opus-4-7",
            "claude-opus-4-6",
            "grok-3-fast",
        ]

    def test_missing_release_date_sorts_last_among_ties(self):
        """A spec without released_at loses the tie-break to any dated spec."""
        dated = _spec("dated", 0.005, 0.025, date(2026, 1, 1))
        undated = _spec("undated", 0.005, 0.025, None)
        ranked = sorted([undated, dated], key=_rank_key, reverse=True)
        assert ranked[0].model_id == "dated"

    def test_model_id_lex_is_final_tiebreak(self):
        """Same cost, same date → higher model_id string wins. Deterministic."""
        a = _spec("aaa", 0.005, 0.025, date(2026, 4, 17))
        z = _spec("zzz", 0.005, 0.025, date(2026, 4, 17))
        ranked = sorted([a, z], key=_rank_key, reverse=True)
        assert ranked[0].model_id == "zzz"
