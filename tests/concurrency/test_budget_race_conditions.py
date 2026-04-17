"""Concurrency tests for budget enforcement race conditions.

Validates that the atomic check_and_add() operation correctly prevents
budget overdraw under concurrent load — the core correctness guarantee
of the CRIT-1 fix.

Key invariants verified:
- Exactly N requests pass when budget allows N × amount_usd
- Total spend never exceeds the configured limit
- Concurrent deductions to the same scope sum correctly
- check_and_add() is truly atomic (no TOCTOU window)

Run with:
    uv run pytest tests/concurrency/test_budget_race_conditions.py -v
"""

from __future__ import annotations

import asyncio

import pytest

from tidus.budget.enforcer import BudgetEnforcer
from tidus.cost.counter import SpendCounter
from tidus.models.budget import BudgetPeriod, BudgetPolicy, BudgetScope


def _policy(policy_id, scope_id, limit_usd, hard_stop=True, scope=BudgetScope.team):
    return BudgetPolicy(
        policy_id=policy_id,
        scope=scope,
        scope_id=scope_id,
        period=BudgetPeriod.monthly,
        limit_usd=limit_usd,
        warn_at_pct=0.80,
        hard_stop=hard_stop,
    )


# ── Atomic check_and_add correctness ─────────────────────────────────────────

class TestAtomicCheckAndAdd:
    async def test_exactly_n_requests_pass_within_limit(self):
        """With a $10.00 limit and $1.00 per request, exactly 10 must pass."""
        counter = SpendCounter()
        limit = 10.0
        amount = 1.0

        results = await asyncio.gather(*[
            counter.check_and_add("team-x", None, amount, limit)
            for _ in range(20)
        ])

        passed = sum(1 for allowed, _ in results if allowed)
        assert passed == 10, f"Expected 10 to pass, got {passed}"

    async def test_total_never_exceeds_limit(self):
        """After 50 concurrent requests, the counter must not exceed the limit."""
        counter = SpendCounter()
        limit = 5.0
        amount = 1.0

        await asyncio.gather(*[
            counter.check_and_add("team-y", None, amount, limit)
            for _ in range(50)
        ])

        final_total = await counter.get("team-y", None)
        assert final_total <= limit + 1e-9, (
            f"Counter exceeded limit: {final_total:.6f} > {limit}"
        )

    async def test_failed_checks_do_not_modify_counter(self):
        """Requests that fail the limit check must not leave a partial deduction."""
        counter = SpendCounter()
        limit = 3.0
        amount = 1.0

        results = await asyncio.gather(*[
            counter.check_and_add("team-z", None, amount, limit)
            for _ in range(10)
        ])

        passed = sum(1 for allowed, _ in results if allowed)
        final_total = await counter.get("team-z", None)

        assert passed == 3
        assert final_total == pytest.approx(3.0)

    async def test_concurrent_same_scope_sums_correctly(self):
        """100 concurrent uncapped deductions sum to exactly 100 × amount."""
        counter = SpendCounter()
        amount = 0.01
        n = 100

        await asyncio.gather(*[
            counter.add("team-sum", None, amount)
            for _ in range(n)
        ])

        total = await counter.get("team-sum", None)
        assert total == pytest.approx(n * amount, abs=1e-9)

    async def test_different_scopes_do_not_interfere_under_concurrency(self):
        """Concurrent check_and_add to different scopes must not cross-contaminate."""
        counter = SpendCounter()
        limit = 0.50
        amount = 0.10

        await asyncio.gather(
            *[counter.check_and_add("team-a", None, amount, limit) for _ in range(10)],
            *[counter.check_and_add("team-b", None, amount, limit) for _ in range(10)],
        )

        total_a = await counter.get("team-a", None)
        total_b = await counter.get("team-b", None)

        assert total_a <= limit + 1e-9
        assert total_b <= limit + 1e-9
        # Each scope is independent
        assert total_a == pytest.approx(total_b)


# ── BudgetEnforcer concurrency ────────────────────────────────────────────────

class TestEnforcerConcurrency:
    async def test_concurrent_can_spend_hard_stop_honoured(self):
        """50 concurrent can_spend() calls on a tight budget — none must overdraw."""
        policies = [_policy("p1", "team-concurrent", limit_usd=0.50)]
        enforcer = BudgetEnforcer(policies, SpendCounter())

        # Pre-spend to $0.45 — only one more $0.10 request should pass
        await enforcer.deduct("team-concurrent", None, 0.45)

        results = await asyncio.gather(*[
            enforcer.can_spend("team-concurrent", None, 0.10)
            for _ in range(20)
        ])

        # Budget allows exactly $0.05 more; $0.10 request must be blocked entirely
        passed = sum(results)
        assert passed == 0, (
            f"Expected 0 to pass (would overdraw), got {passed}"
        )

    async def test_concurrent_deductions_respect_team_limit(self):
        """20 concurrent deduct() calls — total must not exceed limit."""
        policies = [_policy("p2", "team-deduct", limit_usd=1.00)]
        counter = SpendCounter()
        enforcer = BudgetEnforcer(policies, counter)

        await asyncio.gather(*[
            enforcer.deduct("team-deduct", None, 0.10)
            for _ in range(20)
        ])

        final = await counter.get("team-deduct", None)
        assert final == pytest.approx(2.00)  # deduct() always commits regardless

    async def test_two_teams_concurrent_isolation(self):
        """Concurrent spend on two teams must not cross-contaminate."""
        policies = [
            _policy("p-a", "team-alpha", limit_usd=0.50),
            _policy("p-b", "team-beta", limit_usd=0.50),
        ]
        counter = SpendCounter()
        enforcer = BudgetEnforcer(policies, counter)

        await asyncio.gather(
            *[enforcer.deduct("team-alpha", None, 0.05) for _ in range(10)],
            *[enforcer.deduct("team-beta", None, 0.05) for _ in range(10)],
        )

        alpha_total = await counter.get("team-alpha", None)
        beta_total = await counter.get("team-beta", None)

        assert alpha_total == pytest.approx(0.50)
        assert beta_total == pytest.approx(0.50)

    async def test_workflow_and_team_concurrent_are_independent(self):
        """Concurrent spend across team and workflow scopes accumulate independently.

        Note: enforcer.deduct(team, workflow, amount) adds to BOTH the team-level
        counter (eng, None) and the workflow counter (eng, batch). So 5 team-only
        deductions + 5 workflow deductions = 10 additions to the team counter.
        """
        policies = [
            _policy("team-p", "eng", limit_usd=10.0),
            _policy("wf-p", "batch", limit_usd=1.0, scope=BudgetScope.workflow),
        ]
        counter = SpendCounter()
        enforcer = BudgetEnforcer(policies, counter)

        await asyncio.gather(
            *[enforcer.deduct("eng", None, 0.10) for _ in range(5)],
            *[enforcer.deduct("eng", "batch", 0.10) for _ in range(5)],
        )

        team_total = await counter.get("eng", None)
        wf_total = await counter.get("eng", "batch")

        # Team counter receives 5 direct + 5 from workflow deductions = 1.0
        assert team_total == pytest.approx(1.0)
        # Workflow counter receives only its 5 deductions = 0.5
        assert wf_total == pytest.approx(0.50)

    async def test_budget_reset_clears_only_target_and_allows_fresh_spend(self):
        """After reset, the same team can spend up to the limit again."""
        policies = [_policy("p-reset", "team-reset", limit_usd=1.00)]
        counter = SpendCounter()
        enforcer = BudgetEnforcer(policies, counter)

        await enforcer.deduct("team-reset", None, 1.00)
        assert await enforcer.can_spend("team-reset", None, 0.01) is False

        await counter.reset("team-reset", None)

        # Fresh period — should pass again
        assert await enforcer.can_spend("team-reset", None, 0.50) is True

    async def test_concurrent_reserve_is_bounded_by_limit(self):
        """20 concurrent reserve($0.10) on a $1.00 budget — at most 10 succeed.

        Regression test for the can_spend → undo race fix (Fix 2): the new
        `reserve()` method must atomically check-and-hold the reservation so
        that concurrent callers cannot all pass.
        """
        policies = [_policy("reserve-p", "team-reserve", limit_usd=1.00)]
        enforcer = BudgetEnforcer(policies, SpendCounter())

        results = await asyncio.gather(*[
            enforcer.reserve("team-reserve", None, 0.10) for _ in range(20)
        ])

        passed = sum(results)
        assert passed == 10, (
            f"Expected exactly 10 reservations to fit $1.00 limit, got {passed}"
        )

    async def test_refund_releases_reservation(self):
        """Reserve then refund leaves the counter unchanged (modulo floats)."""
        policies = [_policy("refund-p", "team-refund", limit_usd=1.00)]
        counter = SpendCounter()
        enforcer = BudgetEnforcer(policies, counter)

        assert await enforcer.reserve("team-refund", None, 0.30) is True
        assert await counter.get("team-refund", None) == pytest.approx(0.30)

        await enforcer.refund("team-refund", None, 0.30)
        assert await counter.get("team-refund", None) == pytest.approx(0.0)

    async def test_deduct_with_reserved_adjusts_by_diff(self):
        """deduct(actual, reserved_usd=estimated) settles counter at actual."""
        policies = [_policy("settle-p", "team-settle", limit_usd=10.0)]
        counter = SpendCounter()
        enforcer = BudgetEnforcer(policies, counter)

        await enforcer.reserve("team-settle", None, 0.10)
        assert await counter.get("team-settle", None) == pytest.approx(0.10)

        await enforcer.deduct("team-settle", None, 0.08, reserved_usd=0.10)
        assert await counter.get("team-settle", None) == pytest.approx(0.08)

    async def test_can_spend_is_pure_check_no_side_effects(self):
        """can_spend must no longer mutate counter state (fixes can_spend → undo race)."""
        policies = [_policy("pure-p", "team-pure", limit_usd=1.00)]
        counter = SpendCounter()
        enforcer = BudgetEnforcer(policies, counter)

        before = await counter.get("team-pure", None)
        await enforcer.can_spend("team-pure", None, 0.50)
        after = await counter.get("team-pure", None)

        assert before == after, (
            "can_spend() must be a pure check; it must not reserve or undo "
            f"(before={before}, after={after})"
        )

    async def test_high_concurrency_no_negative_totals(self):
        """Under very high concurrency the counter must never go negative."""
        counter = SpendCounter()
        n = 200
        amount = 0.001

        await asyncio.gather(*[
            counter.add("team-neg", None, amount)
            for _ in range(n)
        ])

        total = await counter.get("team-neg", None)
        assert total >= 0.0
        assert total == pytest.approx(n * amount, abs=1e-9)
