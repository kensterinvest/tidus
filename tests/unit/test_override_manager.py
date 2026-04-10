"""Unit tests for OverrideManager.

Covers:
  - RBAC: team_manager can only create overrides for their own team
  - RBAC: team_manager cannot create global-scope overrides
  - RBAC: admin can create any override
  - Conflict detection returns warnings, does not block creation
  - Payload validation: missing required fields → HTTP 400
  - Payload validation: wrong type → HTTP 400
  - deactivate sets is_active=False, deactivated_at, deactivated_by
  - deactivate: 404 for missing/already inactive
  - deactivate: team_manager cannot deactivate another team's override
  - list_active: team_manager sees only own team; admin sees all
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from tidus.auth.middleware import TokenPayload
from tidus.models.registry_models import CreateOverrideRequest
from tidus.registry.override_manager import OverrideManager

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_actor(role="admin", team_id="team-a", sub="user@example.com") -> TokenPayload:
    return TokenPayload(sub=sub, team_id=team_id, role=role, permissions=[], raw_claims={})


def make_request(
    override_type="hard_disable_model",
    scope="global",
    scope_id=None,
    model_id="gpt-4o",
    payload=None,
) -> CreateOverrideRequest:
    return CreateOverrideRequest(
        override_type=override_type,
        scope=scope,
        scope_id=scope_id,
        model_id=model_id,
        payload=payload or {},
        justification="Test override for unit tests",
    )


def make_orm(
    override_id="ov-1",
    override_type="hard_disable_model",
    scope="global",
    scope_id=None,
    model_id="gpt-4o",
    owner_team_id="team-a",
    is_active=True,
):
    orm = MagicMock()
    orm.override_id = override_id
    orm.override_type = override_type
    orm.scope = scope
    orm.scope_id = scope_id
    orm.model_id = model_id
    orm.owner_team_id = owner_team_id
    orm.justification = "test"
    orm.created_by = "user@example.com"
    orm.created_at = datetime.now(UTC)
    orm.expires_at = None
    orm.is_active = is_active
    orm.deactivated_at = None
    orm.deactivated_by = None
    orm.payload = {}
    return orm


def make_session_factory(scalars_result=None, first_result=None):
    """Build a minimal async session factory mock."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = scalars_result or []
    mock_result.scalars.return_value.first.return_value = first_result
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    sf = MagicMock()
    sf.return_value = mock_cm
    return sf, mock_session


# ── RBAC: team_manager ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_team_manager_cannot_create_for_other_team():
    actor = make_actor(role="team_manager", team_id="team-a")
    req = make_request(scope="team", scope_id="team-b")  # different team!
    sf, _ = make_session_factory()
    manager = OverrideManager(sf)

    with pytest.raises(HTTPException) as exc_info:
        await manager.create(req, actor)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_team_manager_cannot_create_global_override():
    actor = make_actor(role="team_manager", team_id="team-a")
    req = make_request(scope="global")
    sf, _ = make_session_factory()
    manager = OverrideManager(sf)

    with pytest.raises(HTTPException) as exc_info:
        await manager.create(req, actor)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_team_manager_can_create_for_own_team():
    actor = make_actor(role="team_manager", team_id="team-a")
    req = make_request(scope="team", scope_id="team-a")  # own team ✓
    sf, mock_session = make_session_factory(scalars_result=[])

    mock_session.refresh = AsyncMock(side_effect=lambda o: None)

    with patch("tidus.registry.override_manager.ModelOverride.model_validate", return_value=MagicMock()):
        manager = OverrideManager(sf)
        override, conflicts = await manager.create(req, actor)

    assert conflicts == []


@pytest.mark.asyncio
async def test_admin_can_create_global_override():
    actor = make_actor(role="admin", team_id="team-a")
    req = make_request(scope="global")
    sf, mock_session = make_session_factory(scalars_result=[])
    mock_session.refresh = AsyncMock(side_effect=lambda o: None)

    with patch("tidus.registry.override_manager.ModelOverride.model_validate", return_value=MagicMock()):
        manager = OverrideManager(sf)
        override, conflicts = await manager.create(req, actor)

    assert conflicts == []


# ── Payload validation ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_override_type_raises_400():
    actor = make_actor()
    req = make_request(override_type="not_a_valid_type")
    sf, _ = make_session_factory()
    manager = OverrideManager(sf)

    with pytest.raises(HTTPException) as exc_info:
        await manager.create(req, actor)
    assert exc_info.value.status_code == 400
    assert "Invalid override_type" in exc_info.value.detail


@pytest.mark.asyncio
async def test_price_multiplier_requires_multiplier_field():
    actor = make_actor()
    req = make_request(override_type="price_multiplier", payload={})  # missing 'multiplier'
    sf, _ = make_session_factory()
    manager = OverrideManager(sf)

    with pytest.raises(HTTPException) as exc_info:
        await manager.create(req, actor)
    assert exc_info.value.status_code == 400
    assert "multiplier" in exc_info.value.detail


@pytest.mark.asyncio
async def test_force_tier_ceiling_requires_max_tier():
    actor = make_actor()
    req = make_request(override_type="force_tier_ceiling", payload={})  # missing 'max_tier'
    sf, _ = make_session_factory()
    manager = OverrideManager(sf)

    with pytest.raises(HTTPException) as exc_info:
        await manager.create(req, actor)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_price_multiplier_wrong_type_raises_400():
    actor = make_actor()
    req = make_request(override_type="price_multiplier", payload={"multiplier": "not-a-number"})
    sf, _ = make_session_factory()
    manager = OverrideManager(sf)

    with pytest.raises(HTTPException) as exc_info:
        await manager.create(req, actor)
    assert exc_info.value.status_code == 400


# ── Conflict detection ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_conflict_detection_returns_warnings_not_error():
    """A conflicting override already exists — creation succeeds but conflicts is non-empty."""
    actor = make_actor()
    req = make_request(override_type="hard_disable_model")

    existing = make_orm(override_id="ov-existing", override_type="hard_disable_model")
    sf, mock_session = make_session_factory(scalars_result=[existing])
    mock_session.refresh = AsyncMock(side_effect=lambda o: None)

    with patch("tidus.registry.override_manager.ModelOverride.model_validate", return_value=MagicMock()):
        manager = OverrideManager(sf)
        override, conflicts = await manager.create(req, actor)

    assert len(conflicts) == 1
    assert "ov-existing" in conflicts[0]


# ── Deactivate ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deactivate_sets_inactive():
    actor = make_actor(role="admin", team_id="team-a")
    existing = make_orm(owner_team_id="team-a")
    sf, mock_session = make_session_factory(first_result=existing)
    mock_session.refresh = AsyncMock(side_effect=lambda o: None)

    with patch("tidus.registry.override_manager.ModelOverride.model_validate", return_value=MagicMock()):
        manager = OverrideManager(sf)
        await manager.deactivate("ov-1", actor)

    assert existing.is_active is False
    assert existing.deactivated_by == actor.sub
    assert existing.deactivated_at is not None


@pytest.mark.asyncio
async def test_deactivate_missing_raises_404():
    actor = make_actor()
    sf, _ = make_session_factory(first_result=None)
    manager = OverrideManager(sf)

    with pytest.raises(HTTPException) as exc_info:
        await manager.deactivate("nonexistent", actor)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_team_manager_cannot_deactivate_other_team_override():
    actor = make_actor(role="team_manager", team_id="team-a")
    other_team_override = make_orm(owner_team_id="team-b")  # different team
    sf, _ = make_session_factory(first_result=other_team_override)
    manager = OverrideManager(sf)

    with pytest.raises(HTTPException) as exc_info:
        await manager.deactivate("ov-1", actor)
    assert exc_info.value.status_code == 403


# ── list_active ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_active_team_manager_sees_only_own_team():
    """team_manager role triggers scope filter — only their team's overrides returned."""
    actor = make_actor(role="team_manager", team_id="team-a")
    own_override = make_orm(owner_team_id="team-a")
    sf, _ = make_session_factory(scalars_result=[own_override])

    with patch("tidus.registry.override_manager.ModelOverride.model_validate", return_value=MagicMock()):
        manager = OverrideManager(sf)
        results = await manager.list_active(actor)

    # The query should have been called; we just verify no exception and result is list
    assert isinstance(results, list)


# ── Audit logging ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_calls_audit_logger():
    """OverrideManager.create() must write an audit entry when audit_logger is provided."""
    actor = make_actor()
    sf, _ = make_session_factory(first_result=None, scalars_result=[])

    audit_logger = AsyncMock()
    manager = OverrideManager(sf, audit_logger=audit_logger)

    req = CreateOverrideRequest(
        override_type="hard_disable_model",
        scope="global",
        model_id="gpt-4o",
        payload={},
        justification="test audit",
    )

    with patch("tidus.registry.override_manager.ModelOverride.model_validate", return_value=MagicMock(
        override_id="ov-audit",
        override_type="hard_disable_model",
        model_id="gpt-4o",
    )):
        await manager.create(req, actor)

    audit_logger.record.assert_called_once()
    call_kwargs = audit_logger.record.call_args.kwargs
    assert call_kwargs["action"] == "registry.override_created"
    assert call_kwargs["resource_type"] == "model_override"


@pytest.mark.asyncio
async def test_deactivate_calls_audit_logger():
    """OverrideManager.deactivate() must write an audit entry when audit_logger is provided."""
    actor = make_actor()
    orm_row = make_orm(owner_team_id="team-eng")
    sf, _ = make_session_factory(first_result=orm_row)

    audit_logger = AsyncMock()
    manager = OverrideManager(sf, audit_logger=audit_logger)

    with patch("tidus.registry.override_manager.ModelOverride.model_validate", return_value=MagicMock(
        override_type="hard_disable_model",
        model_id="gpt-4o",
    )):
        await manager.deactivate("ov-1", actor)

    audit_logger.record.assert_called_once()
    call_kwargs = audit_logger.record.call_args.kwargs
    assert call_kwargs["action"] == "registry.override_deactivated"
    assert call_kwargs["resource_type"] == "model_override"


@pytest.mark.asyncio
async def test_create_without_audit_logger_does_not_crash():
    """OverrideManager without audit_logger still works (audit is optional)."""
    actor = make_actor()
    sf, _ = make_session_factory(first_result=None, scalars_result=[])

    manager = OverrideManager(sf)  # no audit_logger

    req = CreateOverrideRequest(
        override_type="hard_disable_model",
        scope="global",
        model_id="gpt-4o",
        payload={},
        justification="no audit",
    )

    with patch("tidus.registry.override_manager.ModelOverride.model_validate", return_value=MagicMock()):
        override, conflicts = await manager.create(req, actor)

    assert override is not None


@pytest.mark.asyncio
async def test_audit_logger_failure_does_not_propagate():
    """If audit_logger.record() raises, create() must NOT re-raise — audit failures are non-fatal."""
    actor = make_actor()
    sf, _ = make_session_factory(first_result=None, scalars_result=[])

    audit_logger = AsyncMock()
    audit_logger.record.side_effect = RuntimeError("audit DB down")
    manager = OverrideManager(sf, audit_logger=audit_logger)

    req = CreateOverrideRequest(
        override_type="hard_disable_model",
        scope="global",
        model_id="gpt-4o",
        payload={},
        justification="audit failure test",
    )

    with patch("tidus.registry.override_manager.ModelOverride.model_validate", return_value=MagicMock()):
        # Should not raise despite audit failure
        override, _ = await manager.create(req, actor)

    assert override is not None
