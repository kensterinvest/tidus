from datetime import datetime
from pydantic import BaseModel, Field


class CostEstimate(BaseModel):
    """Pre-execution cost estimate for a model + task pair."""

    model_id: str
    vendor: str
    raw_input_tokens: int
    buffered_input_tokens: int
    estimated_output_tokens: int
    buffered_output_tokens: int
    estimated_cost_usd: float
    buffer_pct: float = Field(0.15, description="Safety buffer fraction applied")


class CostRecord(BaseModel):
    """Persisted record of actual cost incurred for a completed task."""

    id: str
    task_id: str
    team_id: str
    workflow_id: str | None
    agent_session_id: str | None
    agent_depth: int
    routing_decision_id: str
    model_id: str
    vendor: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime
    fallback_used: bool = False
    fallback_from: str | None = None


class PriceChangeRecord(BaseModel):
    """Audit record written when the weekly price sync detects a price change."""

    id: str
    model_id: str
    vendor: str
    field_changed: str          # "input_price" | "output_price" | "max_context"
    old_value: float
    new_value: float
    delta_pct: float
    detected_at: datetime
    source: str = "weekly_sync"
