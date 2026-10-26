"""RBAC: catalog-wide / wildcard overrides are admin-only (OVR-2 / OVR-3 / ISO-11).

The EffectiveRegistry is a single global view, so an override that is not pinned
to a specific model (``model_id=None``) or that is a catalog-wide control type
(``emergency_freeze_revision`` / ``pin_provider``) affects routing for EVERY
tenant. Before this gate, a ``team_manager`` could author a team-scoped wildcard
``hard_disable_model`` (→ empties the catalog, 100% routing outage for all teams)
or an ``emergency_freeze_revision`` (→ freezes the whole catalog, silencing even
admin overrides and health telemetry). Those must require ``admin``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from tidus.auth.middleware import TokenPayload
from tidus.models.registry_models import CreateOverrideRequest
from tidus.registry.override_manager import OverrideManager


def make_actor(role="admin", team_id="team-a", sub="user@example.com") -> TokenPayload:
    return TokenPayload(sub=sub, team_id=team_id, role=role, permissions=[], raw_claims={})


def make_request(override_type, *, scope="team", scope_id="team-a", model_id="gpt-4o", payload=None):
    return CreateOverrideRequest(
        override_type=override_type,
        scope=scope,
        scope_id=scope_id,
        model_id=model_id,
        payload=payload or {},
        justification="Test override for unit tests",
    )


def make_session_factory(scalars_result=None):
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = scalars_result or []
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock(side_effect=lambda o: None)
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    sf = MagicMock()
    sf.return_value = mock_cm
    return sf


async def _create(req, actor):
    with patch(
        "tidus.registry.override_manager.ModelOverride.model_validate",
        return_value=MagicMock(),
    ):
        return await OverrideManager(make_session_factory(scalars_result=[])).create(req, actor)


# ── team_manager is BLOCKED from catalog-wide types ─────────────────────────────

@pytest.mark.asyncio
async def test_team_manager_cannot_create_emergency_freeze():
    actor = make_actor(role="team_manager", team_id="team-a")
    # model_id set so ONLY the catalog-wide-type gate can trigger (not the wildcard gate)
    req = make_request("emergency_freeze_revision", model_id="gpt-4o", payload={})
    with pytest.raises(HTTPException) as exc:
        await _create(req, actor)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_team_manager_cannot_create_pin_provider():
    actor = make_actor(role="team_manager", team_id="team-a")
    req = make_request("pin_provider", model_id="gpt-4o", payload={"vendor": "openai"})
    with pytest.raises(HTTPException) as exc:
        await _create(req, actor)
    assert exc.value.status_code == 403


# ── team_manager is BLOCKED from wildcard (model_id=None) overrides ──────────────

@pytest.mark.asyncio
async def test_team_manager_cannot_create_wildcard_hard_disable():
    actor = make_actor(role="team_manager", team_id="team-a")
    req = make_request("hard_disable_model", model_id=None, payload={})
    with pytest.raises(HTTPException) as exc:
        await _create(req, actor)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_team_manager_cannot_create_wildcard_price_multiplier():
    actor = make_actor(role="team_manager", team_id="team-a")
    req = make_request("price_multiplier", model_id=None, payload={"multiplier": 2.0})
    with pytest.raises(HTTPException) as exc:
        await _create(req, actor)
    assert exc.value.status_code == 403


# ── Regression: team_manager can STILL create model-specific overrides ───────────

@pytest.mark.asyncio
async def test_team_manager_can_create_model_specific_hard_disable():
    actor = make_actor(role="team_manager", team_id="team-a")
    req = make_request("hard_disable_model", model_id="gpt-4o", payload={})
    override, conflicts = await _create(req, actor)
    assert conflicts == []


# ── Regression: admin retains full power ────────────────────────────────────────

@pytest.mark.asyncio
async def test_admin_can_create_emergency_freeze():
    actor = make_actor(role="admin", team_id="team-a")
    req = make_request("emergency_freeze_revision", scope="global", scope_id=None, model_id=None, payload={})
    override, conflicts = await _create(req, actor)
    assert conflicts == []


@pytest.mark.asyncio
async def test_admin_can_create_wildcard_hard_disable():
    actor = make_actor(role="admin", team_id="team-a")
    req = make_request("hard_disable_model", scope="global", scope_id=None, model_id=None, payload={})
    override, conflicts = await _create(req, actor)
    assert conflicts == []
