"""Integration tests for BudgetEnforcer warm-start (BUDGET-1).

The in-memory SpendCounter starts empty on every process start. Without a replay
from the persisted cost ledger, a team that exhausted its monthly budget gets a
full reset on every deploy / crash / OOM — defeating the hard-stop. warm_start()
seeds the counter from the CostRecord ledger for each policy's current period so
enforcement survives a restart.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tidus.budget.enforcer import BudgetEnforcer, period_start
from tidus.cost.counter import SpendCounter
from tidus.db.engine import Base
from tidus.db.repositories.cost_repo import CostRepository
from tidus.models.budget import BudgetPeriod, BudgetPolicy, BudgetScope
from tidus.models.cost import CostRecord


@pytest_asyncio.fixture
async def sf():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def _insert_cost(sf, team_id, workflow_id, cost_usd, ts):
    rec = CostRecord(
        id=str(uuid.uuid4()),
        task_id=str(uuid.uuid4()),
        team_id=team_id,
        workflow_id=workflow_id,
        agent_session_id=None,
        agent_depth=0,
        routing_decision_id=str(uuid.uuid4()),
        model_id="gpt-4o",
        vendor="openai",
        input_tokens=100,
        output_tokens=50,
        cost_usd=cost_usd,
        latency_ms=12.0,
        timestamp=ts,
    )
    async with sf() as session:
        await CostRepository(session).insert(rec)


def _team_policy(team_id, limit, hard_stop=True):
    return BudgetPolicy(
        policy_id=f"{team_id}-monthly",
        scope=BudgetScope.team,
        scope_id=team_id,
        period=BudgetPeriod.monthly,
        limit_usd=limit,
        hard_stop=hard_stop,
    )


def _workflow_policy(workflow_id, limit):
    return BudgetPolicy(
        policy_id=f"{workflow_id}-monthly",
        scope=BudgetScope.workflow,
        scope_id=workflow_id,
        period=BudgetPeriod.monthly,
        limit_usd=limit,
        hard_stop=True,
    )


@pytest.mark.asyncio
async def test_warm_start_seeds_team_counter_and_enforces_after_restart(sf):
    """A fresh enforcer warm-started from the ledger reflects in-period spend,
    excludes prior periods, and hard-stops a team already over budget."""
    now = datetime(2026, 6, 12, tzinfo=UTC)
    # In-period (June) → sums to $9.00
    await _insert_cost(sf, "team-eng", None, 5.0, datetime(2026, 6, 2, tzinfo=UTC))
    await _insert_cost(sf, "team-eng", "wf-chat", 4.0, datetime(2026, 6, 10, tzinfo=UTC))
    # Out-of-period (May) → must be excluded
    await _insert_cost(sf, "team-eng", None, 100.0, datetime(2026, 5, 20, tzinfo=UTC))

    enforcer = BudgetEnforcer([_team_policy("team-eng", 10.0)], SpendCounter())
    seeded = await enforcer.warm_start(sf, now=now)

    status = await enforcer.status("team-eng")
    assert status.spent_usd == pytest.approx(9.0)
    assert seeded >= 1
    # $9 of a $10 budget already spent → a $2 request must now be rejected
    assert await enforcer.reserve("team-eng", None, 2.0) is False
    # but a $0.50 request still fits
    assert await enforcer.reserve("team-eng", None, 0.5) is True


@pytest.mark.asyncio
async def test_warm_start_seeds_workflow_counter_per_team(sf):
    """Workflow-scoped policies seed (team_id, workflow_id) counters per team."""
    now = datetime(2026, 6, 12, tzinfo=UTC)
    await _insert_cost(sf, "team-a", "wf-chat", 3.0, datetime(2026, 6, 3, tzinfo=UTC))
    await _insert_cost(sf, "team-b", "wf-chat", 7.0, datetime(2026, 6, 4, tzinfo=UTC))
    await _insert_cost(sf, "team-a", "wf-other", 50.0, datetime(2026, 6, 5, tzinfo=UTC))

    enforcer = BudgetEnforcer([_workflow_policy("wf-chat", 8.0)], SpendCounter())
    await enforcer.warm_start(sf, now=now)

    assert (await enforcer.status("team-a", "wf-chat")).spent_usd == pytest.approx(3.0)
    assert (await enforcer.status("team-b", "wf-chat")).spent_usd == pytest.approx(7.0)
    # team-b is at $7 of $8 → a $2 request on wf-chat is rejected
    assert await enforcer.reserve("team-b", "wf-chat", 2.0) is False


@pytest.mark.asyncio
async def test_warm_start_empty_ledger_is_noop(sf):
    """No ledger rows → counter stays empty, nothing seeded."""
    enforcer = BudgetEnforcer([_team_policy("team-eng", 10.0)], SpendCounter())
    seeded = await enforcer.warm_start(sf, now=datetime(2026, 6, 12, tzinfo=UTC))
    assert seeded == 0
    assert (await enforcer.status("team-eng")).spent_usd == pytest.approx(0.0)


def test_period_start_boundaries():
    """period_start anchors each period correctly (UTC)."""
    now = datetime(2026, 6, 12, 15, 30, tzinfo=UTC)  # a Friday
    assert period_start(BudgetPeriod.daily, now) == datetime(2026, 6, 12, tzinfo=UTC)
    assert period_start(BudgetPeriod.monthly, now) == datetime(2026, 6, 1, tzinfo=UTC)
    # ISO week: 2026-06-12 is a Friday → week began Monday 2026-06-08
    assert period_start(BudgetPeriod.weekly, now) == datetime(2026, 6, 8, tzinfo=UTC)
    # rolling_30d → exactly 30 days before now
    assert period_start(BudgetPeriod.rolling_30d, now) == datetime(2026, 5, 13, 15, 30, tzinfo=UTC)
