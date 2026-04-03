"""GET /api/v1/audit/events — Paginated audit log query (admin only).

Returns a time-descending list of audit events. Supports filtering by
team, action, outcome, and time range for SIEM / compliance tooling.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role, require_role
from tidus.db.engine import AuditLogORM, get_db

router = APIRouter(prefix="/audit", tags=["Audit"])


class AuditEvent(BaseModel):
    id: str
    timestamp: datetime
    actor_team_id: str
    actor_role: str
    actor_sub: str
    action: str
    resource_type: str | None
    resource_id: str | None
    outcome: str
    rejection_reason: str | None
    ip_address: str | None
    metadata: dict[str, Any] | None

    model_config = {"from_attributes": True}


class AuditEventsResponse(BaseModel):
    events: list[AuditEvent]
    total: int
    limit: int
    offset: int


@router.get(
    "/events",
    response_model=AuditEventsResponse,
    summary="List audit events (admin only)",
    response_description="Paginated audit log entries, newest first",
)
async def list_audit_events(
    _auth: Annotated[TokenPayload, Depends(require_role(Role.admin))],
    db: Annotated[AsyncSession, Depends(get_db)],
    team_id: str | None = Query(None, description="Filter by actor team"),
    action: str | None = Query(None, description="Filter by action verb"),
    outcome: str | None = Query(None, description="Filter by outcome (success/rejected/error)"),
    since: datetime | None = Query(None, description="Return events at or after this UTC timestamp"),
    until: datetime | None = Query(None, description="Return events at or before this UTC timestamp"),
    limit: int = Query(100, ge=1, le=1000, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
) -> AuditEventsResponse:
    """Query the audit log with optional filters.

    All parameters are optional — omitting them returns the full log
    (newest first), paginated by ``limit`` / ``offset``.
    """
    stmt = select(AuditLogORM)

    if team_id:
        stmt = stmt.where(AuditLogORM.actor_team_id == team_id)
    if action:
        stmt = stmt.where(AuditLogORM.action == action)
    if outcome:
        stmt = stmt.where(AuditLogORM.outcome == outcome)
    if since:
        stmt = stmt.where(AuditLogORM.timestamp >= since)
    if until:
        stmt = stmt.where(AuditLogORM.timestamp <= until)

    count_result = await db.execute(select(func.count()).select_from(stmt.subquery()))
    total = count_result.scalar_one()

    stmt = stmt.order_by(AuditLogORM.timestamp.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    events = [
        AuditEvent(
            id=row.id,
            timestamp=row.timestamp,
            actor_team_id=row.actor_team_id,
            actor_role=row.actor_role,
            actor_sub=row.actor_sub,
            action=row.action,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            outcome=row.outcome,
            rejection_reason=row.rejection_reason,
            ip_address=row.ip_address,
            metadata=row.metadata_,
        )
        for row in rows
    ]

    return AuditEventsResponse(events=events, total=total, limit=limit, offset=offset)
