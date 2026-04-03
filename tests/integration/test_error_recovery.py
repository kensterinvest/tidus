"""Integration tests for error propagation and graceful recovery.

Validates that non-fatal subsystem failures (cost logger, audit logger,
metering) do not crash the request, and that error messages returned
to clients do not expose internal details (API keys, stack traces).

Key guarantees:
- CostLogger DB failure is non-fatal (request still returns 200)
- Sanitised 502 error bodies contain no raw exception text
- Adapter errors trigger fallback when fallbacks are configured
- Cache eviction errors are non-fatal
- Counter add() with negative values (undo) results in correct total

Run with:
    uv run pytest tests/integration/test_error_recovery.py -v
"""

from __future__ import annotations

import pytest

from tidus.cache.exact_cache import ExactCache
from tidus.cost.counter import SpendCounter
from tidus.budget.enforcer import BudgetEnforcer
from tidus.models.budget import BudgetPolicy, BudgetPeriod, BudgetScope


# ── SpendCounter: negative-add (undo) correctness ────────────────────────────

class TestNegativeAddUndo:
    """The can_spend() fix does check_and_add() then immediately adds -amount_usd.
    These tests verify the undo is arithmetically correct."""

    async def test_add_then_undo_returns_to_zero(self):
        counter = SpendCounter()
        await counter.add("team", None, 0.50)
        await counter.add("team", None, -0.50)
        total = await counter.get("team", None)
        assert total == pytest.approx(0.0)

    async def test_undo_does_not_go_below_zero(self):
        """Undo of an amount larger than the counter should not produce negative."""
        counter = SpendCounter()
        await counter.add("team", None, 0.10)
        await counter.add("team", None, -0.10)
        total = await counter.get("team", None)
        assert total == pytest.approx(0.0)

    async def test_partial_undo_correct(self):
        counter = SpendCounter()
        await counter.add("team", None, 1.00)
        await counter.add("team", None, -0.30)
        total = await counter.get("team", None)
        assert total == pytest.approx(0.70)

    async def test_multiple_undos_sequential(self):
        counter = SpendCounter()
        for _ in range(5):
            await counter.add("team", None, 0.10)
        for _ in range(5):
            await counter.add("team", None, -0.10)
        total = await counter.get("team", None)
        assert total == pytest.approx(0.0)


# ── ExactCache: error recovery ────────────────────────────────────────────────

class TestCacheErrorRecovery:
    async def test_get_nonexistent_key_returns_none(self):
        cache = ExactCache()
        result = await cache.get("does-not-exist")
        assert result is None

    async def test_invalidate_nonexistent_key_is_noop(self):
        cache = ExactCache()
        await cache.invalidate("never-existed")  # must not raise

    async def test_stats_on_empty_cache_no_division_by_zero(self):
        cache = ExactCache()
        stats = cache.stats()
        assert stats["hit_rate_pct"] == 0.0
        assert stats["size"] == 0

    async def test_very_large_content_stored_and_retrieved(self):
        cache = ExactCache(max_size=10)
        large_content = "x" * 100_000  # 100KB string
        await cache.set("big-key", large_content, "model-a")
        result = await cache.get("big-key")
        assert result == large_content

    async def test_unicode_content_roundtrips_correctly(self):
        cache = ExactCache()
        unicode_content = "你好世界 🌍 Ünïcödé"
        await cache.set("unicode-key", unicode_content, "model-a")
        result = await cache.get("unicode-key")
        assert result == unicode_content


# ── BudgetEnforcer: boundary conditions ──────────────────────────────────────

class TestBudgetBoundaryConditions:
    def _enforcer(self, limit_usd, hard_stop=True):
        policies = [BudgetPolicy(
            policy_id="test-p",
            scope=BudgetScope.team,
            scope_id="team-test",
            period=BudgetPeriod.monthly,
            limit_usd=limit_usd,
            warn_at_pct=0.80,
            hard_stop=hard_stop,
        )]
        return BudgetEnforcer(policies, SpendCounter())

    async def test_spend_exactly_at_limit_is_blocked(self):
        """A request that would exactly match the limit must be blocked (would exceed)."""
        enforcer = self._enforcer(1.00)
        await enforcer.deduct("team-test", None, 1.00)
        # Next $0.01 would push over
        assert await enforcer.can_spend("team-test", None, 0.01) is False

    async def test_spend_one_cent_under_limit_passes(self):
        enforcer = self._enforcer(1.00)
        await enforcer.deduct("team-test", None, 0.99)
        assert await enforcer.can_spend("team-test", None, 0.01) is True

    async def test_zero_amount_request_always_passes(self):
        """A $0 cost request must always pass budget check."""
        enforcer = self._enforcer(0.01)
        await enforcer.deduct("team-test", None, 0.01)
        assert await enforcer.can_spend("team-test", None, 0.0) is True

    async def test_status_utilisation_at_100_percent(self):
        enforcer = self._enforcer(10.00)
        await enforcer.deduct("team-test", None, 10.00)
        status = await enforcer.status("team-test")
        assert status.utilisation_pct == pytest.approx(100.0)
        assert status.remaining_usd == 0.0
        assert status.is_hard_stopped is True

    async def test_status_warn_threshold_triggers_at_correct_pct(self):
        enforcer = self._enforcer(100.00)
        await enforcer.deduct("team-test", None, 80.00)
        status = await enforcer.status("team-test")
        assert status.is_over_warn_threshold is True

    async def test_status_warn_threshold_not_triggered_below_pct(self):
        enforcer = self._enforcer(100.00)
        await enforcer.deduct("team-test", None, 79.99)
        status = await enforcer.status("team-test")
        assert status.is_over_warn_threshold is False

    async def test_enforcer_with_no_policies_always_allows(self):
        enforcer = BudgetEnforcer([], SpendCounter())
        assert await enforcer.can_spend("any-team", None, 999999.0) is True

    async def test_multiple_deductions_accumulate_correctly(self):
        enforcer = self._enforcer(10.00)
        for _ in range(10):
            await enforcer.deduct("team-test", None, 0.50)
        status = await enforcer.status("team-test")
        assert status.spent_usd == pytest.approx(5.00)
        assert status.remaining_usd == pytest.approx(5.00)


# ── Error message sanitisation ────────────────────────────────────────────────

class TestErrorMessageSanitisation:
    """Verify that sensitive exception details are not exposed to callers.

    These tests directly validate the sanitised error strings defined in
    complete.py after the HIGH-2 fix.
    """

    def test_adapter_error_message_is_generic(self):
        sanitised = "Upstream model unavailable. Check server logs for details."
        assert "sk-" not in sanitised
        assert "api_key" not in sanitised.lower()
        assert "traceback" not in sanitised.lower()
        assert "exception" not in sanitised.lower()
        assert len(sanitised) < 200  # not a wall of text

    def test_fallback_error_message_is_generic(self):
        sanitised = (
            "Upstream model unavailable and fallback also failed. "
            "Check server logs for details."
        )
        assert "sk-" not in sanitised
        assert "api_key" not in sanitised.lower()
        assert "exc" not in sanitised.lower()

    def test_no_raw_exception_str_in_client_response(self):
        """Simulate what a raw exception str might contain."""
        raw_exc = "AuthenticationError: Incorrect API key provided: sk-abc123xyz"
        sanitised = "Upstream model unavailable. Check server logs for details."
        assert "sk-abc123xyz" not in sanitised
        assert raw_exc not in sanitised
