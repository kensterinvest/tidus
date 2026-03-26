from datetime import datetime
from pydantic import BaseModel, Field


class GuardrailPolicy(BaseModel):
    """Limits applied to every agent session to prevent runaway compute.

    Example:
        policy = GuardrailPolicy(max_agent_depth=5, max_tokens_per_step=8000)
    """

    max_agent_depth: int = Field(5, ge=1, description="Maximum agent recursion depth")
    max_tokens_per_step: int = Field(8000, ge=1, description="Max input+output tokens per call")
    max_retries_per_task: int = Field(3, ge=0, description="Max retries before hard rejection")
    max_parallel_sessions_per_team: int = Field(10, ge=1)


class AgentSession(BaseModel):
    """Tracks the state of a running multi-agent session.

    Example:
        session = AgentSession(
            session_id="...",
            team_id="team-engineering",
            started_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
        )
    """

    session_id: str
    team_id: str
    workflow_id: str | None = None
    current_depth: int = Field(0, ge=0)
    total_tokens_used: int = Field(0, ge=0)
    retry_count: int = Field(0, ge=0)
    started_at: datetime
    last_active: datetime
    is_active: bool = True
