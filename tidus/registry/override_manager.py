"""OverrideManager — CRUD for model overrides with RBAC and conflict detection.

RBAC rules enforced here:
  - team_manager: can only create/delete overrides where scope_id == their team_id
  - admin: unrestricted create/delete

Conflict detection:
  Before inserting, queries for existing active overrides on the same
  (model_id, override_type, scope_id) tuple. Conflicts are allowed — they coexist
  via the deterministic precedence table in merge.py — but a `conflicts` list is
  returned so the operator is informed.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import HTTPException, status
from sqlalchemy import select

from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role
from tidus.db.registry_orm import ModelOverrideORM
from tidus.models.registry_models import (
    VALID_OVERRIDE_TYPES,
    CreateOverrideRequest,
    ModelOverride,
)

log = structlog.get_logger(__name__)


class OverrideManager:
    """Create, list, and deactivate model overrides with RBAC enforcement."""

    def __init__(self, session_factory, audit_logger=None) -> None:
        self._sf = session_factory
        self._audit = audit_logger

    # ── Create ────────────────────────────────────────────────────────────────

    async def create(
        self,
        request: CreateOverrideRequest,
        actor: TokenPayload,
    ) -> tuple[ModelOverride, list[str]]:
        """Create an override and return (override, conflicts_list).

        Raises HTTP 400 on invalid override_type or payload.
        Raises HTTP 403 when team_manager tries to create an override for another team.
        """
        # Validate override type
        if request.override_type not in VALID_OVERRIDE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid override_type: {request.override_type!r}. "
                       f"Valid types: {sorted(VALID_OVERRIDE_TYPES)}",
            )

        # RBAC scope check: team_manager can only manage their own team's overrides
        actor_role = Role(actor.role) if hasattr(Role, actor.role) else None
        if actor_role == Role.team_manager:
            if request.scope == "team" and request.scope_id != actor.team_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="team_manager may only create overrides scoped to their own team",
                )
            if request.scope == "global":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="team_manager may only create team-scoped overrides; global scope requires admin",
                )

        # Payload validation per type
        _validate_payload(request.override_type, request.payload)

        override_id = str(uuid.uuid4())

        async with self._sf() as session:
            # Conflict detection
            conflicts = await _find_conflicts(
                session,
                override_type=request.override_type,
                model_id=request.model_id,
                scope_id=request.scope_id,
            )

            orm = ModelOverrideORM(
                override_id=override_id,
                override_type=request.override_type,
                scope=request.scope,
                scope_id=request.scope_id,
                model_id=request.model_id,
                payload=request.payload,
                owner_team_id=actor.team_id,
                justification=request.justification,
                created_by=actor.sub,
                expires_at=request.expires_at,
                is_active=True,
            )
            session.add(orm)
            await session.commit()
            await session.refresh(orm)

            override = ModelOverride.model_validate(orm)

        conflict_msgs = [
            f"Conflicts with existing {c.override_type} override {c.override_id} "
            f"(model_id={c.model_id!r}, scope_id={c.scope_id!r})"
            for c in conflicts
        ]
        if conflict_msgs:
            log.warning(
                "override_conflict_detected",
                override_id=override_id,
                conflicts=len(conflict_msgs),
            )

        log.info(
            "override_created",
            override_id=override_id,
            override_type=request.override_type,
            model_id=request.model_id,
            created_by=actor.sub,
        )
        if self._audit is not None:
            try:
                await self._audit.record(
                    actor=actor,
                    action="registry.override_created",
                    resource_type="model_override",
                    resource_id=override_id,
                    metadata={
                        "override_type": request.override_type,
                        "model_id": request.model_id,
                        "scope": request.scope,
                        "scope_id": request.scope_id,
                        "conflicts": len(conflict_msgs),
                    },
                )
            except Exception as exc:
                log.warning("override_audit_failed", error=str(exc))
        return override, conflict_msgs

    # ── List ──────────────────────────────────────────────────────────────────

    async def list_active(self, actor: TokenPayload) -> list[ModelOverride]:
        """List active overrides. team_manager sees only their team's overrides."""
        async with self._sf() as session:
            q = select(ModelOverrideORM).where(ModelOverrideORM.is_active == True)  # noqa: E712

            actor_role = Role(actor.role) if hasattr(Role, actor.role) else None
            if actor_role == Role.team_manager:
                q = q.where(ModelOverrideORM.owner_team_id == actor.team_id)

            result = await session.execute(q)
            rows = result.scalars().all()

        return [ModelOverride.model_validate(r) for r in rows]

    # ── Deactivate ────────────────────────────────────────────────────────────

    async def deactivate(self, override_id: str, actor: TokenPayload) -> ModelOverride:
        """Deactivate an override. Returns the updated record.

        Raises HTTP 404 if not found or already inactive.
        Raises HTTP 403 if team_manager tries to deactivate another team's override.
        """
        now = datetime.now(UTC)

        async with self._sf() as session:
            result = await session.execute(
                select(ModelOverrideORM).where(
                    ModelOverrideORM.override_id == override_id,
                    ModelOverrideORM.is_active == True,  # noqa: E712
                )
            )
            orm = result.scalars().first()

            if orm is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Override {override_id!r} not found or already inactive",
                )

            actor_role = Role(actor.role) if hasattr(Role, actor.role) else None
            if actor_role == Role.team_manager and orm.owner_team_id != actor.team_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="team_manager may only deactivate overrides owned by their team",
                )

            orm.is_active = False
            orm.deactivated_at = now
            orm.deactivated_by = actor.sub
            await session.commit()
            await session.refresh(orm)

            override = ModelOverride.model_validate(orm)

        log.info("override_deactivated", override_id=override_id, deactivated_by=actor.sub)
        if self._audit is not None:
            try:
                await self._audit.record(
                    actor=actor,
                    action="registry.override_deactivated",
                    resource_type="model_override",
                    resource_id=override_id,
                    metadata={"override_type": override.override_type, "model_id": override.model_id},
                )
            except Exception as exc:
                log.warning("override_audit_failed", error=str(exc))
        return override


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _find_conflicts(
    session,
    override_type: str,
    model_id: str | None,
    scope_id: str | None,
) -> list[ModelOverrideORM]:
    """Find existing active overrides that overlap with the new one."""
    q = select(ModelOverrideORM).where(
        ModelOverrideORM.is_active == True,  # noqa: E712
        ModelOverrideORM.override_type == override_type,
    )
    if model_id is not None:
        q = q.where(ModelOverrideORM.model_id == model_id)
    if scope_id is not None:
        q = q.where(ModelOverrideORM.scope_id == scope_id)
    result = await session.execute(q)
    return result.scalars().all()


_PAYLOAD_RULES: dict[str, dict[str, Any]] = {
    "price_multiplier":          {"required": ["multiplier"], "types": {"multiplier": (int, float)}},
    "hard_disable_model":        {"required": [], "types": {}},
    "force_tier_ceiling":        {"required": ["max_tier"], "types": {"max_tier": int}},
    "force_local_only":          {"required": [], "types": {}},
    "pin_provider":              {"required": ["vendor"], "types": {"vendor": str}},
    "emergency_freeze_revision": {"required": [], "types": {}},
}


def _validate_payload(override_type: str, payload: dict) -> None:
    rules = _PAYLOAD_RULES.get(override_type, {})
    for field in rules.get("required", []):
        if field not in payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"override_type={override_type!r} requires payload field {field!r}",
            )
    for field, expected_type in rules.get("types", {}).items():
        if field in payload and not isinstance(payload[field], expected_type):
            if isinstance(expected_type, tuple):
                type_name = " or ".join(t.__name__ for t in expected_type)
            else:
                type_name = expected_type.__name__
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"payload[{field!r}] must be {type_name}",
            )
