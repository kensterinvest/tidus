"""Tenant isolation on cross-tenant-leaking endpoints (auth review P1-C, P1-B).

Before this fix, any `read_only`/`developer` token could read EVERY team's
financials (usage/summary, budgets list, monthly report), and `advance_session`
had no team-ownership check (cross-team guardrail DoS). Non-admin/non-team_manager
callers must be scoped to their own team; `advance_session` must mirror the
ownership guard already on get/terminate.

Handlers are tested as plain async functions (Depends injected directly),
matching the existing OverrideManager unit-test style.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from tidus.api.v1.budgets import list_budgets
from tidus.api.v1.guardrails import AdvanceRequest, advance_session
from tidus.api.v1.reports import monthly_savings_report
from tidus.api.v1.usage import usage_summary
from tidus.budget.enforcer import BudgetEnforcer
from tidus.cost.counter import SpendCounter
from tidus.models.budget import BudgetPeriod, BudgetPolicy, BudgetScope


def _auth(role, team_id="team-a", sub="u@example.com"):
    from tidus.auth.middleware import TokenPayload
    return TokenPayload(sub=sub, team_id=team_id, role=role, permissions=[], raw_claims={})


def _enforcer():
    policies = [
        BudgetPolicy(policy_id="a", scope=BudgetScope.team, scope_id="team-a",
                     period=BudgetPeriod.monthly, limit_usd=100.0),
        BudgetPolicy(policy_id="b", scope=BudgetScope.team, scope_id="team-b",
                     period=BudgetPeriod.monthly, limit_usd=100.0),
    ]
    return BudgetEnforcer(policies, SpendCounter())


# ── usage_summary ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_usage_summary_readonly_scoped_to_own_team():
    results = await usage_summary(_enforcer(), _auth("read_only", "team-a"), team_id=None)
    assert {r.team_id for r in results} == {"team-a"}


@pytest.mark.asyncio
async def test_usage_summary_readonly_explicit_other_team_403():
    with pytest.raises(HTTPException) as exc:
        await usage_summary(_enforcer(), _auth("developer", "team-a"), team_id="team-b")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_usage_summary_admin_sees_all():
    results = await usage_summary(_enforcer(), _auth("admin", "team-a"), team_id=None)
    assert {r.team_id for r in results} == {"team-a", "team-b"}


# ── list_budgets ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_budgets_readonly_scoped_to_own_team():
    policies = await list_budgets(_enforcer(), _auth("read_only", "team-a"))
    assert {p.scope_id for p in policies} == {"team-a"}


@pytest.mark.asyncio
async def test_list_budgets_admin_sees_all():
    policies = await list_budgets(_enforcer(), _auth("admin", "team-a"))
    assert {p.scope_id for p in policies} == {"team-a", "team-b"}


# ── monthly_savings_report ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_report_developer_explicit_other_team_403():
    with pytest.raises(HTTPException) as exc:
        await monthly_savings_report(
            MagicMock(), _auth("developer", "team-a"),
            year=2026, month=5, team_id="team-b",
        )
    assert exc.value.status_code == 403


# ── advance_session ──────────────────────────────────────────────────────────────

def _store_returning(team_id):
    store = MagicMock()
    store.get = AsyncMock(return_value=MagicMock(team_id=team_id))
    return store


@pytest.mark.asyncio
async def test_advance_session_cross_team_404():
    guard = MagicMock()
    guard.check_and_advance = AsyncMock(return_value=MagicMock(allowed=True, reason=None))
    store = _store_returning("team-b")  # session belongs to team-b
    with pytest.raises(HTTPException) as exc:
        await advance_session(
            AdvanceRequest(session_id="s1", input_tokens=10),
            guard, store, _auth("developer", "team-a"),
        )
    assert exc.value.status_code == 404
    guard.check_and_advance.assert_not_called()


@pytest.mark.asyncio
async def test_advance_session_own_team_allowed():
    guard = MagicMock()
    guard.check_and_advance = AsyncMock(return_value=MagicMock(allowed=True, reason=None))
    store = _store_returning("team-a")  # own team
    out = await advance_session(
        AdvanceRequest(session_id="s1", input_tokens=10),
        guard, store, _auth("developer", "team-a"),
    )
    assert out["allowed"] is True
    guard.check_and_advance.assert_awaited_once()
