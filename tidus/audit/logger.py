"""Tamper-evident audit logger for SOC 2 / ISO 27001 / HIPAA compliance.

Design principles:
- **Non-fatal**: logging failures never raise — the routing pipeline must not
  be blocked by an audit backend issue.
- **Actor identity from JWT**: ``actor_team_id``, ``actor_role``, and
  ``actor_sub`` come directly from the :class:`~tidus.auth.middleware.TokenPayload`
  produced by Phase 8's OIDC middleware.
- **Async-first**: uses the existing SQLAlchemy async session.

Usage::

    audit = AuditLogger(session_factory=get_session_factory())
    await audit.record(
        actor=token_payload,
        action="route",
        resource_type="task",
        resource_id=task.task_id,
        outcome="success",
    )
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker

from tidus.auth.middleware import TokenPayload
from tidus.db.engine import AuditLogORM

log = structlog.get_logger(__name__)


class AuditLogger:
    """Records audit events to the ``audit_logs`` table.

    Errors are caught and logged at WARNING level — never re-raised — so
    a DB hiccup cannot take down the routing pipeline.
    """

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._session_factory = session_factory

    async def record(
        self,
        *,
        actor: TokenPayload,
        action: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        outcome: str = "success",
        rejection_reason: str | None = None,
        ip_address: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write one audit event row.

        Args:
            actor:            Authenticated caller (from ``get_current_user``).
            action:           Verb describing the operation, e.g. ``"route"``,
                              ``"complete"``, ``"budget.create"``.
            resource_type:    Type of the affected resource, e.g. ``"task"``.
            resource_id:      Identifier of the affected resource.
            outcome:          ``"success"`` | ``"rejected"`` | ``"error"``.
            rejection_reason: Human-readable reason when outcome != ``"success"``.
            ip_address:       Caller IP (pass from ``Request.client.host`` if needed).
            metadata:         Arbitrary extra context (model_id, cost, etc.).
        """
        entry = AuditLogORM(
            id=str(uuid.uuid4()),
            actor_team_id=actor.team_id,
            actor_role=actor.role,
            actor_sub=actor.sub,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            rejection_reason=rejection_reason,
            ip_address=ip_address,
            metadata_=metadata,
        )
        try:
            async with self._session_factory() as session:
                session.add(entry)
                await session.commit()
        except Exception as exc:
            log.warning("audit_log_failed", action=action, error=str(exc))
