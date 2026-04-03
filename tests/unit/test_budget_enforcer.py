"""Unit tests for BudgetEnforcer — can_spend() and deduct() logic."""

from __future__ import annotations

import pytest

from tidus.budget.enforcer import BudgetEnforcer
from tidus.cost.counter import SpendCounter
from tidus.models.budget import BudgetPeriod, BudgetPolicy, BudgetScope


def _policy(
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
def team_policy():
    return _policy("team-eng", BudgetScope.team, "team-engineering", limit_usd=10.0)


@pytest.fixture
def enforcer(team_policy):
    return BudgetEnforcer([team_policy], SpendCounter())


# ── can_spend ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_can_spend_within_limit(enforcer):
    assert await enforcer.can_spend("team-engineering", None, 5.0) is True


@pytest.mark.asyncio
async def test_can_spend_exactly_at_limit(enforcer):
    # Spending exactly the full limit should be allowed (not yet exceeded)
    assert await enforcer.can_spend("team-engineering", None, 10.0) is True


@pytest.mark.asyncio
async def test_cannot_spend_above_limit(enforcer):
    assert await enforcer.can_spend("team-engineering", None, 10.01) is False


@pytest.mark.asyncio
async def test_no_policy_always_allows():
    """Without a policy, any spend is allowed."""
    enforcer = BudgetEnforcer([], SpendCounter())
    assert await enforcer.can_spend("any-team", None, 999.0) is True


@pytest.mark.asyncio
async def test_warn_only_policy_always_allows():
    """hard_stop=False policies emit warnings but never block."""
    policy = _policy("warn-only", BudgetScope.team, "team-warn", limit_usd=1.0, hard_stop=False)
    enforcer = BudgetEnforcer([policy], SpendCounter())
    assert await enforcer.can_spend("team-warn", None, 100.0) is True


# ── deduct ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deduct_accumulates(enforcer):
    await enforcer.deduct("team-engineering", None, 3.0)
    await enforcer.deduct("team-engineering", None, 2.0)
    status = await enforcer.status("team-engineering")
    assert status.spent_usd == pytest.approx(5.0)
    assert status.remaining_usd == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_deduct_then_cannot_spend(enforcer):
    """After spending most of the budget, a large request should be blocked."""
    await enforcer.deduct("team-engineering", None, 9.5)
    assert await enforcer.can_spend("team-engineering", None, 1.0) is False
    assert await enforcer.can_spend("team-engineering", None, 0.49) is True


# ── workflow budget ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_workflow_budget_blocks_independently():
    team_pol = _policy("team", BudgetScope.team, "eng", limit_usd=100.0)
    wf_pol = _policy("wf", BudgetScope.workflow, "wf-batch", limit_usd=1.0)
    enforcer = BudgetEnforcer([team_pol, wf_pol], SpendCounter())

    # Workflow budget exceeded even though team budget is fine
    assert await enforcer.can_spend("eng", "wf-batch", 1.01) is False
    assert await enforcer.can_spend("eng", "wf-batch", 0.99) is True


# ── status ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_status_utilisation_pct(enforcer):
    await enforcer.deduct("team-engineering", None, 8.0)
    status = await enforcer.status("team-engineering")
    assert status.utilisation_pct == pytest.approx(80.0)
    assert status.is_over_warn_threshold is True
    assert status.is_hard_stopped is False


@pytest.mark.asyncio
async def test_status_hard_stopped_when_at_limit(enforcer):
    await enforcer.deduct("team-engineering", None, 10.0)
    status = await enforcer.status("team-engineering")
    assert status.is_hard_stopped is True
