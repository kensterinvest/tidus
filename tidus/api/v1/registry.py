"""Registry API router — revision management, override CRUD, drift events.

Prefix: /api/v1/registry (registered in main.py)

Endpoints:
  GET  /revisions                       — list all revisions (summary)
  GET  /revisions/{id}                  — revision detail + entry count
  POST /revisions/{id}/activate         — admin rollback (re-promote SUPERSEDED)
  POST /revisions/{id}/force-activate   — admin force-promote (bypass Tier 3)
  GET  /overrides                       — list active overrides (team-scoped for managers)
  POST /overrides                       — create override
  GET  /overrides/export                — HMAC-SHA256 signed YAML bundle (admin)
  DELETE /overrides/{id}                — deactivate override
  GET  /drift                           — list open drift events
  POST /drift/{id}/resolve              — mark drift event resolved (developer+)
  GET  /revisions/{id}/diff             — field-level diff vs another revision
  GET  /revisions/{id}/preview          — merged EffectiveModelSpec preview
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import Annotated, Any

import structlog
import yaml
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import select, update

from tidus.api.deps import get_audit_logger, get_override_manager, get_registry, get_session_factory
from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role, require_role
from tidus.db.registry_orm import (
    ModelCatalogRevisionORM,
    ModelDriftEventORM,
)
from tidus.db.repositories.registry_repo import (
    count_entries_all_revisions,
    count_entries_for_revision,
    get_active_overrides,
    get_all_revisions,
    get_entries_for_revision,
    get_open_drift_events,
    get_revision_by_id,
)
from tidus.models.model_registry import ModelSpec
from tidus.models.registry_models import (
    CreateOverrideRequest,
    CreateOverrideResponse,
    ForceActivateRequest,
    ModelOverride,
    ResolveRequest,
    RevisionDetail,
    RevisionDiffEntry,
    RevisionSummary,
)
from tidus.registry.merge import merge_spec
from tidus.registry.telemetry_reader import TelemetryReader
from tidus.settings import get_settings

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/registry", tags=["Registry (Admin)"])


# ── Revisions ─────────────────────────────────────────────────────────────────

@router.get(
    "/revisions",
    response_model=list[RevisionSummary],
    summary="List all catalog revisions",
)
async def list_revisions(
    _: Annotated[TokenPayload, Depends(require_role(Role.read_only, Role.developer, Role.team_manager, Role.admin))],
    session_factory=Depends(get_session_factory),
):
    """Return all catalog revisions ordered by creation time (newest first)."""
    revisions = await get_all_revisions(session_factory)
    # Single GROUP BY query instead of N+1 COUNT-per-revision queries.
    counts = await count_entries_all_revisions(session_factory)
    return [
        RevisionSummary(
            revision_id=rev.revision_id,
            created_at=rev.created_at,
            activated_at=rev.activated_at,
            source=rev.source,
            status=rev.status,
            entry_count=counts.get(rev.revision_id, 0),
        )
        for rev in revisions
    ]


@router.get(
    "/revisions/{revision_id}",
    response_model=RevisionDetail,
    summary="Get revision detail",
)
async def get_revision(
    revision_id: str,
    _: Annotated[TokenPayload, Depends(require_role(Role.read_only, Role.developer, Role.team_manager, Role.admin))],
    session_factory=Depends(get_session_factory),
):
    """Return full detail for a specific revision including canary_results."""
    rev = await get_revision_by_id(session_factory, revision_id)
    if rev is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Revision {revision_id!r} not found")
    count = await count_entries_for_revision(session_factory, revision_id)
    return RevisionDetail(
        revision_id=rev.revision_id,
        created_at=rev.created_at,
        activated_at=rev.activated_at,
        source=rev.source,
        signature_hash=rev.signature_hash,
        status=rev.status,
        failure_reason=rev.failure_reason,
        canary_results=rev.canary_results,
        entry_count=count,
    )


@router.post(
    "/revisions/{revision_id}/activate",
    response_model=RevisionDetail,
    summary="Rollback: re-promote a SUPERSEDED revision to ACTIVE",
)
async def activate_revision(
    revision_id: str,
    actor: Annotated[TokenPayload, Depends(require_role(Role.admin))],
    session_factory=Depends(get_session_factory),
    registry=Depends(get_registry),
    audit_logger=Depends(get_audit_logger),
):
    """Atomically promote a SUPERSEDED revision back to ACTIVE.

    The current ACTIVE revision is set to SUPERSEDED in the same transaction.
    Useful for fast rollback when a new revision introduced a regression.
    """
    rev = await get_revision_by_id(session_factory, revision_id)
    if rev is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Revision {revision_id!r} not found")
    if rev.status != "superseded":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Only SUPERSEDED revisions can be re-activated. Current status: {rev.status!r}",
        )

    now = datetime.now(UTC)
    async with session_factory() as session:
        # Two-phase atomic flip: ACTIVE → SUPERSEDED, target → ACTIVE
        await session.execute(
            update(ModelCatalogRevisionORM)
            .where(ModelCatalogRevisionORM.status == "active")
            .values(status="superseded")
        )
        await session.execute(
            update(ModelCatalogRevisionORM)
            .where(ModelCatalogRevisionORM.revision_id == revision_id)
            .values(status="active", activated_at=now)
        )
        await session.commit()

    log.info("revision_rollback_activated", revision_id=revision_id, actor=actor.sub)
    await audit_logger.record(
        actor=actor,
        action="registry.revision_activate",
        resource_type="catalog_revision",
        resource_id=revision_id,
        outcome="success",
    )

    # Invalidate the in-memory cache immediately
    if hasattr(registry, "refresh"):
        try:
            await registry.refresh(session_factory)
        except Exception as exc:
            log.error("revision_activate_refresh_failed", error=str(exc))

    rev = await get_revision_by_id(session_factory, revision_id)
    count = await count_entries_for_revision(session_factory, revision_id)
    return RevisionDetail(
        revision_id=rev.revision_id,
        created_at=rev.created_at,
        activated_at=rev.activated_at,
        source=rev.source,
        signature_hash=rev.signature_hash,
        status=rev.status,
        failure_reason=rev.failure_reason,
        canary_results=rev.canary_results,
        entry_count=count,
    )


@router.post(
    "/revisions/{revision_id}/force-activate",
    response_model=RevisionDetail,
    summary="Force-promote a revision (bypasses Tier 3 canary)",
)
async def force_activate_revision(
    revision_id: str,
    body: ForceActivateRequest,
    actor: Annotated[TokenPayload, Depends(require_role(Role.admin))],
    session_factory=Depends(get_session_factory),
    registry=Depends(get_registry),
    audit_logger=Depends(get_audit_logger),
):
    """Force-promote a SUPERSEDED or PENDING revision, bypassing only Tier 3 canary.

    Tier 1 (schema) and Tier 2 (invariant) validation still run via
    RegistryPipeline.force_activate().  A mandatory justification is required
    and written to the audit log.
    """
    from tidus.registry.pipeline import RegistryPipeline

    pipeline = RegistryPipeline(session_factory, registry)
    try:
        await pipeline.force_activate(revision_id, actor, body.justification)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    log.info(
        "revision_force_activated",
        revision_id=revision_id,
        actor=actor.sub,
        justification=body.justification,
    )
    await audit_logger.record(
        actor=actor,
        action="registry.force_promote",
        resource_type="catalog_revision",
        resource_id=revision_id,
        outcome="success",
        metadata={"justification": body.justification},
    )

    rev = await get_revision_by_id(session_factory, revision_id)
    count = await count_entries_for_revision(session_factory, revision_id)
    return RevisionDetail(
        revision_id=rev.revision_id,
        created_at=rev.created_at,
        activated_at=rev.activated_at,
        source=rev.source,
        signature_hash=rev.signature_hash,
        status=rev.status,
        failure_reason=rev.failure_reason,
        canary_results=rev.canary_results,
        entry_count=count,
    )


@router.get(
    "/revisions/{revision_id}/diff",
    response_model=list[RevisionDiffEntry],
    summary="Field-level diff between two revisions",
)
async def diff_revisions(
    _: Annotated[TokenPayload, Depends(require_role(Role.read_only, Role.developer, Role.team_manager, Role.admin))],
    revision_id: str,
    base: str = Query(..., description="Revision ID to compare against"),
    session_factory=Depends(get_session_factory),
):
    """Compare all model specs between revision_id and base revision.

    Returns only models where at least one field changed, with {from, to} per field.
    """
    rev = await get_revision_by_id(session_factory, revision_id)
    if rev is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Revision {revision_id!r} not found")
    base_rev = await get_revision_by_id(session_factory, base)
    if base_rev is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Base revision {base!r} not found")

    new_entries = await get_entries_for_revision(session_factory, revision_id)
    old_entries = await get_entries_for_revision(session_factory, base)

    old_by_model: dict[str, dict] = {e.model_id: e.spec_json for e in old_entries}
    new_by_model: dict[str, dict] = {e.model_id: e.spec_json for e in new_entries}

    all_model_ids = old_by_model.keys() | new_by_model.keys()
    diffs = []
    for model_id in sorted(all_model_ids):
        old_spec = old_by_model.get(model_id, {})
        new_spec = new_by_model.get(model_id, {})
        changed_fields: dict[str, dict[str, Any]] = {}
        for field in old_spec.keys() | new_spec.keys():
            old_val = old_spec.get(field)
            new_val = new_spec.get(field)
            if old_val != new_val:
                changed_fields[field] = {"from": old_val, "to": new_val}
        if changed_fields:
            diffs.append(RevisionDiffEntry(model_id=model_id, changed_fields=changed_fields))

    return diffs


@router.get(
    "/revisions/{revision_id}/preview",
    response_model=list[dict],
    summary="Preview merged EffectiveModelSpec for a revision",
)
async def preview_revision(
    _: Annotated[TokenPayload, Depends(require_role(Role.read_only, Role.developer, Role.team_manager, Role.admin))],
    revision_id: str,
    session_factory=Depends(get_session_factory),
):
    """Show the full merged EffectiveModelSpec for each model as it would appear
    if this revision were promoted to ACTIVE.

    Combines the revision's base entries with the current active overrides and
    current telemetry. Useful for validating a SUPERSEDED or PENDING revision
    before force-activating it.
    """
    rev = await get_revision_by_id(session_factory, revision_id)
    if rev is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Revision {revision_id!r} not found")

    entries = await get_entries_for_revision(session_factory, revision_id)
    overrides = await get_active_overrides(session_factory)
    telemetry_map = await TelemetryReader().get_all_snapshots(session_factory)

    result = []
    for entry in entries:
        try:
            base = ModelSpec.model_validate(entry.spec_json)
        except Exception:
            continue
        telemetry = telemetry_map.get(base.model_id)
        merged = merge_spec(base, overrides, telemetry)
        result.append(merged.model_dump())

    return result


# ── Overrides ─────────────────────────────────────────────────────────────────

@router.get(
    "/overrides",
    response_model=list[ModelOverride],
    summary="List active overrides",
)
async def list_overrides(
    actor: Annotated[TokenPayload, Depends(require_role(Role.team_manager, Role.admin))],
    override_manager=Depends(get_override_manager),
):
    """List active overrides. team_manager sees only their own team's overrides."""
    return await override_manager.list_active(actor)


@router.post(
    "/overrides",
    response_model=CreateOverrideResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an override",
)
async def create_override(
    body: CreateOverrideRequest,
    actor: Annotated[TokenPayload, Depends(require_role(Role.team_manager, Role.admin))],
    override_manager=Depends(get_override_manager),
):
    """Create an override and return it along with any conflict warnings.

    Conflicts are allowed — they coexist via the deterministic precedence table.
    The conflicts list is informational, not an error.
    """
    override, conflicts = await override_manager.create(body, actor)
    return CreateOverrideResponse(override=override, conflicts=conflicts)


@router.get(
    "/overrides/export",
    summary="Export active overrides as a signed YAML bundle (admin)",
)
async def export_overrides(
    actor: Annotated[TokenPayload, Depends(require_role(Role.admin))],
    session_factory=Depends(get_session_factory),
):
    """Export all active overrides as an HMAC-SHA256 signed YAML bundle.

    When TIDUS_REGISTRY_EXPORT_SIGNING_KEY is set, the response includes an
    X-Tidus-Signature header with the HMAC of the YAML body. Useful for GitOps
    workflows where the bundle is stored in version control.
    """
    settings = get_settings()
    overrides = await get_active_overrides(session_factory)

    bundle = {
        "exported_at": datetime.now(UTC).isoformat(),
        "exported_by": actor.sub,
        "overrides": [
            {
                "override_id": o.override_id,
                "override_type": o.override_type,
                "scope": o.scope,
                "scope_id": o.scope_id,
                "model_id": o.model_id,
                "payload": o.payload,
                "owner_team_id": o.owner_team_id,
                "justification": o.justification,
                "created_by": o.created_by,
                "created_at": o.created_at.isoformat() if o.created_at else None,
                "expires_at": o.expires_at.isoformat() if o.expires_at else None,
            }
            for o in overrides
        ],
    }

    yaml_body = yaml.dump(bundle, default_flow_style=False, allow_unicode=True)
    yaml_bytes = yaml_body.encode("utf-8")

    headers: dict[str, str] = {"Content-Type": "application/yaml"}
    signing_key = settings.tidus_registry_export_signing_key
    if signing_key:
        sig = hmac.new(signing_key.encode(), yaml_bytes, hashlib.sha256).hexdigest()
        headers["X-Tidus-Signature"] = f"hmac-sha256={sig}"
    else:
        log.warning("registry_export_unsigned", reason="TIDUS_REGISTRY_EXPORT_SIGNING_KEY not set")

    return Response(content=yaml_bytes, headers=headers, media_type="application/yaml")


@router.delete(
    "/overrides/{override_id}",
    response_model=ModelOverride,
    summary="Deactivate an override",
)
async def deactivate_override(
    override_id: str,
    actor: Annotated[TokenPayload, Depends(require_role(Role.team_manager, Role.admin))],
    override_manager=Depends(get_override_manager),
):
    """Deactivate an override. team_manager can only deactivate their own team's overrides."""
    return await override_manager.deactivate(override_id, actor)


# ── Drift events ──────────────────────────────────────────────────────────────

@router.get(
    "/drift",
    response_model=list[dict],
    summary="List open drift events",
)
async def list_drift_events(
    _: Annotated[TokenPayload, Depends(require_role(Role.read_only, Role.developer, Role.team_manager, Role.admin))],
    session_factory=Depends(get_session_factory),
):
    """Return all open (unresolved) model drift events."""
    events = await get_open_drift_events(session_factory)
    return [
        {
            "id": e.id,
            "model_id": e.model_id,
            "drift_type": e.drift_type,
            "severity": e.severity,
            "detected_at": e.detected_at.isoformat() if e.detected_at else None,
            "metric_value": e.metric_value,
            "threshold_value": e.threshold_value,
            "drift_status": e.drift_status,
            "active_revision_id": e.active_revision_id,
        }
        for e in events
    ]


@router.post(
    "/drift/{event_id}/resolve",
    response_model=dict,
    summary="Manually resolve a drift event",
)
async def resolve_drift_event(
    event_id: str,
    body: ResolveRequest,
    actor: Annotated[TokenPayload, Depends(require_role(Role.developer, Role.team_manager, Role.admin))],
    session_factory=Depends(get_session_factory),
):
    """Mark a drift event as manually resolved.

    The status check and mutation happen in a single session to eliminate
    the TOCTOU race between checking drift_status and writing the update.
    """
    now = datetime.now(UTC)
    async with session_factory() as session:
        result = await session.execute(
            select(ModelDriftEventORM).where(ModelDriftEventORM.id == event_id)
        )
        orm = result.scalars().first()
        if orm is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Drift event {event_id!r} not found",
            )
        if orm.drift_status != "open":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Drift event is already resolved (status={orm.drift_status!r})",
            )
        orm.drift_status = "manually_resolved"
        orm.resolved_at = now
        await session.commit()
        await session.refresh(orm)

    log.info("drift_event_resolved", event_id=event_id, resolved_by=actor.sub, notes=body.resolution_notes)
    return {
        "id": orm.id,
        "drift_status": orm.drift_status,
        "resolved_at": orm.resolved_at.isoformat() if orm.resolved_at else None,
        "resolution_notes": body.resolution_notes,
    }
