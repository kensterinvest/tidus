from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class BudgetPeriod(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    rolling_30d = "rolling_30d"


class BudgetScope(str, Enum):
    team = "team"
    workflow = "workflow"


class BudgetPolicy(BaseModel):
    """Defines a spending limit for a team or workflow.

    Example:
        policy = BudgetPolicy(
            policy_id="team-eng-monthly",
            scope=BudgetScope.team,
            scope_id="team-engineering",
            period=BudgetPeriod.monthly,
            limit_usd=500.0,
        )
    """

    policy_id: str
    scope: BudgetScope
    scope_id: str = Field(..., description="The team_id or workflow_id this policy governs")
    period: BudgetPeriod
    limit_usd: float = Field(..., gt=0)
    warn_at_pct: float = Field(0.80, ge=0.0, le=1.0, description="Fraction at which to emit a warning alert")
    hard_stop: bool = Field(True, description="True = reject requests when limit exceeded; False = warn only")


class BudgetStatus(BaseModel):
    """Live snapshot of a budget policy's current utilisation."""

    policy_id: str
    scope: BudgetScope
    scope_id: str
    period: BudgetPeriod
    limit_usd: float
    spent_usd: float
    remaining_usd: float
    utilization_pct: float
    is_exceeded: bool
    is_warning: bool
    period_start: datetime
    period_end: datetime
