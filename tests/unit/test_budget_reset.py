"""Unit tests for BudgetEnforcer.reset_period().

Verifies that the monthly reset correctly clears counters for matching
policies and leaves non-matching policies untouched.
"""

from __future__ import annotations

import pytest

from tidus.budget.enforcer import BudgetEnforcer
from tidus.cost.counter import SpendCounter
from tidus.models.budget import BudgetPeriod, BudgetPolicy, BudgetScope


def _policy(policy_id, scope_id, period, limit_usd=100.0):
    return BudgetPolicy(
        policy_id=policy_id,
        scope=BudgetScope.team,
        scope_id=scope_id,
        period=period,
        limit_usd=limit_usd,
        warn_at_pct=0.80,
        hard_stop=True,
    )


class TestBudgetPeriodReset:
    async def test_monthly_reset_clears_monthly_counters(self):
        """reset_period(monthly) must zero out monthly-scoped team counters."""
        counter = SpendCounter()
        enforcer = BudgetEnforcer(
            [_policy("p1", "team-a", BudgetPeriod.monthly)], counter
        )
        await enforcer.deduct("team-a", None, 75.0)

        count = await enforcer.reset_period(BudgetPeriod.monthly)
        assert count == 1

        status = await enforcer.status("team-a")
        assert status.spent_usd == pytest.approx(0.0)

    async def test_monthly_reset_does_not_touch_daily_counters(self):
        """reset_period(monthly) must leave daily-period counters unchanged."""
        counter = SpendCounter()
        enforcer = BudgetEnforcer(
            [
                _policy("m1", "team-monthly", BudgetPeriod.monthly),
                _policy("d1", "team-daily",   BudgetPeriod.daily),
            ],
            counter,
        )
        await enforcer.deduct("team-monthly", None, 50.0)
        await enforcer.deduct("team-daily",   None, 10.0)

        count = await enforcer.reset_period(BudgetPeriod.monthly)
        assert count == 1

        monthly_status = await enforcer.status("team-monthly")
        daily_status   = await enforcer.status("team-daily")

        assert monthly_status.spent_usd == pytest.approx(0.0)
        assert daily_status.spent_usd   == pytest.approx(10.0)

    async def test_reset_multiple_monthly_policies(self):
        """reset_period(monthly) resets all monthly teams in one call."""
        counter = SpendCounter()
        policies = [
            _policy("p-eng", "team-eng",    BudgetPeriod.monthly),
            _policy("p-mkt", "team-mkt",    BudgetPeriod.monthly),
            _policy("p-dat", "team-data",   BudgetPeriod.monthly),
        ]
        enforcer = BudgetEnforcer(policies, counter)

        await enforcer.deduct("team-eng",  None, 100.0)
        await enforcer.deduct("team-mkt",  None, 50.0)
        await enforcer.deduct("team-data", None, 200.0)

        count = await enforcer.reset_period(BudgetPeriod.monthly)
        assert count == 3

        for team in ("team-eng", "team-mkt", "team-data"):
            s = await enforcer.status(team)
            assert s.spent_usd == pytest.approx(0.0), f"{team} was not reset"

    async def test_reset_allows_spend_again_after_reset(self):
        """After a reset, a previously hard-stopped team can spend again."""
        counter = SpendCounter()
        enforcer = BudgetEnforcer(
            [_policy("p1", "team-x", BudgetPeriod.monthly, limit_usd=10.0)],
            counter,
        )
        await enforcer.deduct("team-x", None, 10.0)
        assert await enforcer.can_spend("team-x", None, 0.01) is False

        await enforcer.reset_period(BudgetPeriod.monthly)

        assert await enforcer.can_spend("team-x", None, 5.0) is True

    async def test_reset_on_no_policies_returns_zero(self):
        """reset_period on an enforcer with no policies returns 0."""
        enforcer = BudgetEnforcer([], SpendCounter())
        count = await enforcer.reset_period(BudgetPeriod.monthly)
        assert count == 0

    async def test_workflow_scoped_reset_clears_workflow_counters(self):
        """Fix 9 regression: resetting a workflow-period policy must clear the
        (team_id, workflow_id) counter — not `(workflow_id, None)` which is
        a non-existent team counter."""
        counter = SpendCounter()
        wf_policy = BudgetPolicy(
            policy_id="wf-reset",
            scope=BudgetScope.workflow,
            scope_id="wf-batch",
            period=BudgetPeriod.daily,
            limit_usd=1.0,
            warn_at_pct=0.80,
            hard_stop=True,
        )
        enforcer = BudgetEnforcer([wf_policy], counter)

        # Spend on two teams using the same workflow
        await enforcer.deduct("team-a", "wf-batch", 0.30)
        await enforcer.deduct("team-b", "wf-batch", 0.25)

        count = await enforcer.reset_period(BudgetPeriod.daily)
        assert count == 1

        # Both team-level workflow counters must be zero
        assert await counter.get("team-a", "wf-batch") == pytest.approx(0.0)
        assert await counter.get("team-b", "wf-batch") == pytest.approx(0.0)

    async def test_workflow_reset_does_not_touch_team_counters(self):
        """Resetting a workflow policy must not wipe unrelated team counters."""
        counter = SpendCounter()
        wf_policy = BudgetPolicy(
            policy_id="wf-iso",
            scope=BudgetScope.workflow,
            scope_id="wf-iso-flow",
            period=BudgetPeriod.daily,
            limit_usd=1.0,
            warn_at_pct=0.80,
            hard_stop=True,
        )
        enforcer = BudgetEnforcer([wf_policy], counter)

        await enforcer.deduct("team-a", None, 0.50)  # team-level only
        await enforcer.deduct("team-a", "wf-iso-flow", 0.20)  # touches both
        # After setup: team-a/None = 0.70, team-a/wf-iso-flow = 0.20

        await enforcer.reset_period(BudgetPeriod.daily)

        # Workflow counter reset; team counter untouched (still 0.70)
        assert await counter.get("team-a", "wf-iso-flow") == pytest.approx(0.0)
        assert await counter.get("team-a", None) == pytest.approx(0.70)

    async def test_reset_is_idempotent(self):
        """Calling reset twice in a row must not error and counter stays at 0."""
        counter = SpendCounter()
        enforcer = BudgetEnforcer(
            [_policy("p1", "team-idem", BudgetPeriod.monthly)], counter
        )
        await enforcer.deduct("team-idem", None, 30.0)
        await enforcer.reset_period(BudgetPeriod.monthly)
        await enforcer.reset_period(BudgetPeriod.monthly)  # second call

        s = await enforcer.status("team-idem")
        assert s.spent_usd == pytest.approx(0.0)
