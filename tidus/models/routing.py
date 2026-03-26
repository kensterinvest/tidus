from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RejectionReason(str, Enum):
    """Reason a model candidate was rejected during the 5-stage selection."""

    # Stage 1 — hard constraints
    model_disabled = "model_disabled"           # enabled=False or deprecated=True
    context_too_large = "context_too_large"     # estimated_input_tokens > max_context
    domain_not_supported = "domain_not_supported"  # capability not in spec.capabilities
    privacy_violation = "privacy_violation"     # confidential task routed to cloud model

    # Stage 2 — guardrails
    agent_depth_exceeded = "agent_depth_exceeded"
    token_limit_exceeded = "token_limit_exceeded"

    # Stage 3 — complexity tier ceiling
    complexity_ceiling = "complexity_ceiling"

    # Stage 4 — budget
    budget_exceeded = "budget_exceeded"

    # Catch-all
    no_capable_model = "no_capable_model"


class RoutingDecision(BaseModel):
    """The output of the model selector — either a chosen model or a rejection.

    A decision is accepted when rejection_reason is None.

    Example (success):
        decision = RoutingDecision(
            task_id="task-abc",
            chosen_model_id="claude-haiku-4-5",
            rejection_reason=None,
            score=0.12,
            estimated_cost_usd=0.0008,
        )
    Example (rejection):
        decision = RoutingDecision(
            task_id="task-abc",
            chosen_model_id="gpt-5",
            rejection_reason=RejectionReason.budget_exceeded,
            score=None,
            estimated_cost_usd=0.045,
        )
    """

    task_id: str
    chosen_model_id: str  # model that was selected or that was rejected
    rejection_reason: Optional[RejectionReason] = None
    score: Optional[float] = None           # normalised score (lower = better)
    estimated_cost_usd: Optional[float] = None
    fallback_from: Optional[str] = None     # set when this is a fallback decision

    @property
    def accepted(self) -> bool:
        """True if this decision selects a model (not a rejection)."""
        return self.rejection_reason is None
