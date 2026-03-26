import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Complexity(str, Enum):
    simple = "simple"
    moderate = "moderate"
    complex = "complex"
    critical = "critical"


class Domain(str, Enum):
    chat = "chat"
    code = "code"
    reasoning = "reasoning"
    extraction = "extraction"
    classification = "classification"
    summarization = "summarization"
    creative = "creative"


class Privacy(str, Enum):
    public = "public"
    internal = "internal"
    confidential = "confidential"


class TaskDescriptor(BaseModel):
    """Describes an AI task to be routed through Tidus.

    Example:
        task = TaskDescriptor(
            team_id="team-engineering",
            complexity=Complexity.simple,
            domain=Domain.chat,
            privacy=Privacy.internal,
            estimated_input_tokens=200,
            messages=[{"role": "user", "content": "Summarise this ticket."}],
        )
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    team_id: str = Field(..., description="Team making the request")
    workflow_id: str | None = Field(None, description="Optional workflow identifier")
    agent_session_id: str | None = Field(None, description="Active agent session (for guardrail tracking)")
    agent_depth: int = Field(default=0, ge=0, description="Current recursion depth in an agent chain")

    # Classification signals used by the model selector
    complexity: Complexity = Field(..., description="Task complexity tier")
    domain: Domain = Field(..., description="Primary task domain")
    privacy: Privacy = Field(Privacy.internal, description="Data sensitivity level")

    # Token estimates (used for cost calculation; actual counts logged post-call)
    estimated_input_tokens: int = Field(..., gt=0, description="Estimated input token count")
    estimated_output_tokens: int = Field(256, gt=0, description="Estimated output token count")

    # Optional caller-supplied overrides
    preferred_model_id: str | None = Field(None, description="Bypass routing and use this model")
    max_cost_usd: float | None = Field(None, gt=0, description="Hard ceiling on cost for this request")
    require_streaming: bool = False

    # Payload
    system_prompt: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list, description="OpenAI-compatible message list")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Caller metadata passed through to logs")


class TaskResult(BaseModel):
    """Result returned after a task is routed and executed.

    Example:
        result = TaskResult(task_id="...", model_id="claude-haiku-4-5", ...)
    """

    task_id: str
    routing_decision_id: str
    model_id: str
    vendor: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    response: dict[str, Any]  # normalised to {role, content} shape
    fallback_used: bool = False
    fallback_from: str | None = None
