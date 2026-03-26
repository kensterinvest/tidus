from enum import Enum
from pydantic import BaseModel, Field

from tidus.models.cost import CostEstimate


class RejectionReason(str, Enum):
    budget_exceeded = "budget_exceeded"
    no_capable_model = "no_capable_model"
    guardrail_depth = "guardrail_depth"
    guardrail_tokens = "guardrail_tokens"
    guardrail_retries = "guardrail_retries"
    privacy_constraint = "privacy_constraint"
    all_models_unavailable = "all_models_unavailable"


class RoutingDecision(BaseModel):
    """The output of the model selector — either a chosen model or a rejection.

    Example (success):
        decision = RoutingDecision(
            decision_id="...",
            task_id="...",
            selected_model_id="claude-haiku-4-5",
            selected_vendor="anthropic",
            ...
        )
    Example (rejection):
        decision = RoutingDecision(
            decision_id="...",
            task_id="...",
            selected_model_id=None,
            rejection_reason=RejectionReason.budget_exceeded,
            explanation="Team budget exceeded for period monthly",
        )
    """

    decision_id: str
    task_id: str

    # Set when a model is selected; None when rejected
    selected_model_id: str | None = None
    selected_vendor: str | None = None

    candidates_considered: list[str] = Field(default_factory=list, description="model_ids evaluated")
    rejection_reason: RejectionReason | None = None
    cost_estimates: list[CostEstimate] = Field(default_factory=list)
    chosen_estimate: CostEstimate | None = None
    explanation: str = Field("", description="Human-readable routing rationale")
    budget_status_snapshot: dict = Field(default_factory=dict)

    # Populated when routing falls back to a secondary model
    fallback_from: str | None = None

    @property
    def accepted(self) -> bool:
        return self.selected_model_id is not None
