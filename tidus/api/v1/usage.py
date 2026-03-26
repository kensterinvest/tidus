"""GET /api/v1/usage — Cost usage reporting.

Endpoints:
    GET /api/v1/usage/summary   — Aggregate spend by team (from in-memory counters)

Note: Phase 4 adds per-model breakdown once CostRecord is written to the DB
after each completed call. This endpoint uses the in-memory SpendCounter which
is fast but resets on restart.

Example curl:
    curl "http://localhost:8000/api/v1/usage/summary?team_id=team-engineering"
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from tidus.api.deps import get_enforcer
from tidus.budget.enforcer import BudgetEnforcer
from tidus.models.budget import BudgetStatus

router = APIRouter(prefix="/usage", tags=["Usage"])


class UsageSummary(BaseModel):
    team_id: str
    current_spend_usd: float
    limit_usd: float | None
    utilisation_pct: float | None
    is_hard_stopped: bool


@router.get(
    "/summary",
    response_model=list[UsageSummary],
    summary="Aggregate spend summary for all teams with active policies",
)
async def usage_summary(
    enforcer: Annotated[BudgetEnforcer, Depends(get_enforcer)],
    team_id: str | None = None,
) -> list[UsageSummary]:
    """Return spend vs limit for all teams (or a specific team) with budget policies.

    Teams without a budget policy are not included — their spend is uncapped.
    """
    policies = enforcer.list_policies()
    team_ids = {p.scope_id for p in policies}

    if team_id is not None:
        team_ids = {team_id} & team_ids

    results: list[UsageSummary] = []
    for tid in sorted(team_ids):
        status: BudgetStatus | None = await enforcer.status(team_id=tid)
        if status is None:
            continue
        results.append(
            UsageSummary(
                team_id=tid,
                current_spend_usd=status.spent_usd,
                limit_usd=status.limit_usd,
                utilisation_pct=status.utilisation_pct,
                is_hard_stopped=status.is_hard_stopped,
            )
        )
    return results
