"""Pydantic models for the v1.1.0 registry API layer.

These are the request/response shapes for /api/v1/registry endpoints.
Internal merge logic uses ModelSpec from tidus.models.model_registry.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ── Override models ───────────────────────────────────────────────────────────

VALID_OVERRIDE_TYPES = frozenset({
    "price_multiplier",
    "hard_disable_model",
    "force_tier_ceiling",
    "force_local_only",
    "pin_provider",
    "emergency_freeze_revision",
})


class CreateOverrideRequest(BaseModel):
    override_type: str = Field(..., description="One of: " + ", ".join(sorted(VALID_OVERRIDE_TYPES)))
    scope: Literal["global", "team"] = "global"
    scope_id: str | None = Field(None, description="Team ID when scope='team'")
    model_id: str | None = Field(None, description="Specific model, or None for all models in scope")
    payload: dict[str, Any] = Field(default_factory=dict, description="Type-specific parameters")
    justification: str = Field(..., min_length=5, description="Reason for this override (audit trail)")
    expires_at: datetime | None = Field(None, description="Auto-deactivation timestamp (UTC)")


class ModelOverride(BaseModel):
    override_id: str
    override_type: str
    scope: str
    scope_id: str | None
    model_id: str | None
    payload: dict[str, Any]
    owner_team_id: str
    justification: str
    created_by: str
    created_at: datetime
    expires_at: datetime | None
    is_active: bool
    deactivated_at: datetime | None
    deactivated_by: str | None

    model_config = {"from_attributes": True}


class CreateOverrideResponse(BaseModel):
    override: ModelOverride
    conflicts: list[str] = Field(default_factory=list, description="Warnings about conflicting active overrides")


# ── Revision models ───────────────────────────────────────────────────────────

class RevisionSummary(BaseModel):
    revision_id: str
    created_at: datetime
    activated_at: datetime | None
    source: str
    status: str
    entry_count: int

    model_config = {"from_attributes": True}


class RevisionDetail(BaseModel):
    revision_id: str
    created_at: datetime
    activated_at: datetime | None
    source: str
    signature_hash: str
    status: str
    failure_reason: str | None
    canary_results: Any | None
    entry_count: int

    model_config = {"from_attributes": True}


class ForceActivateRequest(BaseModel):
    justification: str = Field(..., min_length=10, description="Mandatory reason for bypassing Tier 3 canary")


class RevisionDiffEntry(BaseModel):
    model_id: str
    changed_fields: dict[str, dict[str, Any]]  # {field: {"from": old, "to": new}}


class ResolveRequest(BaseModel):
    resolution_notes: str = Field(..., min_length=5)


# ── Telemetry snapshot (used internally by TelemetryReader + merge) ───────────

class TelemetrySnapshot(BaseModel):
    model_id: str
    measured_at: datetime
    latency_p50_ms: int | None
    is_healthy: bool
    consecutive_failures: int
    staleness: Literal["fresh", "unknown", "expired"]

    model_config = {"from_attributes": True}
