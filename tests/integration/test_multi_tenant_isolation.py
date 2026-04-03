"""Integration tests for multi-tenant isolation in Tidus.

Verifies that budget spend, budget enforcement, and routing decisions
for one team cannot bleed into another team's state.

Key guarantees tested:
- SpendCounter isolates (team_id, workflow_id) pairs independently
- BudgetEnforcer hard-stops only the team that exceeded its limit
- A team with no policy is never blocked by another team's exhaustion
- Warn-only policies never block even when budget is depleted
- Workflow-scoped budgets are isolated from team-level counters

Run with:
    uv run pytest tests/integration/test_multi_tenant_isolation.py -v
"""

from __future__ import annotations

import pytest

from tidus.budget.enforcer import BudgetEnforcer
from tidus.cost.counter import SpendCounter
from tidus.models.budget import BudgetPeriod, BudgetPolicy, BudgetScope

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_policy(
    policy_id: str,
    scope: BudgetScope,
    scope_id: str,
    limit_usd: float,
    hard_stop: bool = True,
    warn_at_pct: float = 0.80,
) -> BudgetPolicy:
    return BudgetPolicy(
        policy_id=policy_id,
        scope=scope,
        scope_id=scope_id,
        period=BudgetPeriod.monthly,
        limit_usd=limit_usd,
        warn_at_pct=warn_at_pct,
        hard_stop=hard_stop,
    )


@pytest.fixture
def two_team_enforcer():
    """Enforcer with two separate team policies and a fresh counter."""
    policies = [
        _make_policy("eng-policy", BudgetScope.team, "team-engineering", limit_usd=10.0),
        _make_policy("mkt-policy", BudgetScope.team, "team-marketing", limit_usd=5.0),
    ]
    return BudgetEnforcer(policies, SpendCounter())


# ── Budget spend isolation ─────────────────────────────────────────────────────

class TestSpendIsolation:
    async def test_team_a_spend_does_not_affect_team_b(self, two_team_enforcer):
        """Deducting from team-engineering must not increase team-marketing spend."""
        await two_team_enforcer.deduct("team-engineering", None, 9.0)

        status_mkt = await two_team_enforcer.status("team-marketing")
        assert status_mkt.spent_usd == 0.0
        assert status_mkt.remaining_usd == pytest.approx(5.0)

    async def test_team_b_spend_does_not_affect_team_a(self, two_team_enforcer):
        """Deducting from team-marketing must not increase team-engineering spend."""
        await two_team_enforcer.deduct("team-marketing", None, 5.0)

        status_eng = await two_team_enforcer.status("team-engineering")
        assert status_eng.spent_usd == 0.0
        assert status_eng.remaining_usd == pytest.approx(10.0)

    async def test_concurrent_deductions_do_not_cross_contaminate(self, two_team_enforcer):
        """Multiple deductions from both teams accumulate independently."""
        await two_team_enforcer.deduct("team-engineering", None, 3.0)
        await two_team_enforcer.deduct("team-marketing", None, 2.0)
        await two_team_enforcer.deduct("team-engineering", None, 2.0)
        await two_team_enforcer.deduct("team-marketing", None, 1.0)

        eng = await two_team_enforcer.status("team-engineering")
        mkt = await two_team_enforcer.status("team-marketing")

        assert eng.spent_usd == pytest.approx(5.0)
        assert mkt.spent_usd == pytest.approx(3.0)


# ── Hard-stop isolation ────────────────────────────────────────────────────────

class TestHardStopIsolation:
    async def test_team_a_exhausted_does_not_block_team_b(self, two_team_enforcer):
        """When team-engineering hits its limit, team-marketing must still be spendable."""
        await two_team_enforcer.deduct("team-engineering", None, 10.0)

        # team-engineering is now at limit
        assert await two_team_enforcer.can_spend("team-engineering", None, 0.01) is False
        # team-marketing is unaffected
        assert await two_team_enforcer.can_spend("team-marketing", None, 1.0) is True

    async def test_both_teams_exhausted_independently(self, two_team_enforcer):
        """Both teams can reach their own limits without affecting each other."""
        await two_team_enforcer.deduct("team-engineering", None, 10.0)
        await two_team_enforcer.deduct("team-marketing", None, 5.0)

        assert await two_team_enforcer.can_spend("team-engineering", None, 0.01) is False
        assert await two_team_enforcer.can_spend("team-marketing", None, 0.01) is False

    async def test_team_without_policy_always_passes(self, two_team_enforcer):
        """A team with no configured budget policy is never blocked."""
        await two_team_enforcer.deduct("team-engineering", None, 10.0)

        # Unknown team has no policy → always allowed
        assert await two_team_enforcer.can_spend("team-unknown", None, 999.0) is True

    async def test_warn_only_team_never_hard_stops(self):
        """A warn-only policy (hard_stop=False) never blocks even at 1000% utilisation."""
        policies = [
            _make_policy("warn-mkt", BudgetScope.team, "team-marketing", limit_usd=1.0, hard_stop=False),
        ]
        enforcer = BudgetEnforcer(policies, SpendCounter())
        await enforcer.deduct("team-marketing", None, 100.0)

        assert await enforcer.can_spend("team-marketing", None, 999.0) is True


# ── Workflow-scope isolation ───────────────────────────────────────────────────

class TestWorkflowScopeIsolation:
    async def test_workflow_budget_exhaustion_does_not_block_team(self):
        """Workflow budget exhaustion blocks the workflow scope, not the team."""
        policies = [
            _make_policy("team-eng", BudgetScope.team, "eng", limit_usd=100.0),
            _make_policy("wf-batch", BudgetScope.workflow, "batch-job", limit_usd=1.0),
        ]
        enforcer = BudgetEnforcer(policies, SpendCounter())
        await enforcer.deduct("eng", "batch-job", 1.0)

        # batch-job workflow is exhausted
        assert await enforcer.can_spend("eng", "batch-job", 0.01) is False
        # team-eng as a whole is fine
        assert await enforcer.can_spend("eng", None, 50.0) is True

    async def test_different_workflows_isolated_from_each_other(self):
        """Workflow-A spend does not affect workflow-B counters."""
        policies = [
            _make_policy("team", BudgetScope.team, "eng", limit_usd=100.0),
            _make_policy("wf-a", BudgetScope.workflow, "workflow-a", limit_usd=5.0),
            _make_policy("wf-b", BudgetScope.workflow, "workflow-b", limit_usd=5.0),
        ]
        counter = SpendCounter()
        enforcer = BudgetEnforcer(policies, counter)

        await enforcer.deduct("eng", "workflow-a", 5.0)

        wf_b_total = await counter.get("eng", "workflow-b")
        assert wf_b_total == 0.0
        assert await enforcer.can_spend("eng", "workflow-b", 4.99) is True


# ── SpendCounter isolation (unit-level) ──────────────────────────────────────

class TestSpendCounterIsolation:
    async def test_different_team_keys_are_independent(self):
        counter = SpendCounter()
        await counter.add("team-a", None, 10.0)
        await counter.add("team-b", None, 5.0)

        assert await counter.get("team-a", None) == pytest.approx(10.0)
        assert await counter.get("team-b", None) == pytest.approx(5.0)

    async def test_team_plus_workflow_key_separate_from_team_only(self):
        """(team, workflow) counter does not contribute to (team, None) counter."""
        counter = SpendCounter()
        await counter.add("eng", "wf-x", 7.0)

        assert await counter.get("eng", None) == 0.0
        assert await counter.get("eng", "wf-x") == pytest.approx(7.0)

    async def test_reset_only_clears_target_scope(self):
        counter = SpendCounter()
        await counter.add("team-a", None, 10.0)
        await counter.add("team-b", None, 8.0)

        await counter.reset("team-a", None)

        assert await counter.get("team-a", None) == 0.0
        assert await counter.get("team-b", None) == pytest.approx(8.0)
