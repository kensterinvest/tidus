"""GET/POST /api/v1/budgets — Budget policy management.

Endpoints:
    GET    /api/v1/budgets                          — List active budget policies
    POST   /api/v1/budgets                          — Create a new budget policy
    GET    /api/v1/budgets/status/team/{team_id}    — Live spend vs limit for a team

Example curl:
    curl http://localhost:8000/api/v1/budgets/status/team/team-engineering
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from tidus.api.deps import get_enforcer
from tidus.budget.enforcer import BudgetEnforcer
from tidus.models.budget import BudgetPeriod, BudgetPolicy, BudgetScope, BudgetStatus

router = APIRouter(prefix="/budgets", tags=["Budgets"])


# ── Request model ─────────────────────────────────────────────────────────────

class CreateBudgetRequest(BaseModel):
    policy_id: str = Field(..., description="Unique identifier for this policy")
    scope: BudgetScope
    scope_id: str = Field(..., description="Team or workflow ID this policy applies to")
    period: BudgetPeriod
    limit_usd: float = Field(..., gt=0, description="Spend limit in USD")
    warn_at_pct: float = Field(0.80, ge=0, le=1.0)
    hard_stop: bool = True


# ── Route handlers ────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=list[BudgetPolicy],
    summary="List all active budget policies",
)
async def list_budgets(
    enforcer: Annotated[BudgetEnforcer, Depends(get_enforcer)],
) -> list[BudgetPolicy]:
    return enforcer.list_policies()


@router.post(
    "",
    response_model=BudgetPolicy,
    status_code=201,
    summary="Create a new budget policy",
)
async def create_budget(
    body: CreateBudgetRequest,
    enforcer: Annotated[BudgetEnforcer, Depends(get_enforcer)],
) -> BudgetPolicy:
    """Add a new budget policy to the in-memory registry.

    This does NOT persist to budgets.yaml — it takes effect immediately
    for all subsequent routing decisions until the service restarts.
    """
    policy = BudgetPolicy(
        policy_id=body.policy_id,
        scope=body.scope,
        scope_id=body.scope_id,
        period=body.period,
        limit_usd=body.limit_usd,
        warn_at_pct=body.warn_at_pct,
        hard_stop=body.hard_stop,
    )
    enforcer.add_policy(policy)
    return policy


@router.get(
    "/status/team/{team_id}",
    response_model=BudgetStatus,
    summary="Get live spend vs limit for a team",
)
async def get_team_budget_status(
    team_id: str,
    enforcer: Annotated[BudgetEnforcer, Depends(get_enforcer)],
) -> BudgetStatus:
    """Return current spend, limit, and utilisation for the given team.

    Returns a 404 if no budget policy exists for this team.
    """
    has_policy = any(
        p.scope_id == team_id for p in enforcer.list_policies()
    )
    if not has_policy:
        raise HTTPException(
            status_code=404,
            detail=f"No budget policy found for team '{team_id}'",
        )
    return await enforcer.status(team_id=team_id)
