"""Auth/metering identity hardening (AUTH-2, ISO-9, ISO-8).

AUTH-2: dev-mode (no OIDC) was gated by an exact `environment == "production"`
string, so any other value (staging, test, a typo) silently granted unauthenticated
admin. Now only an explicit dev-environment allowlist enables dev-mode; anything
else fails closed.

ISO-9 / ISO-8: the metering middleware reads `request.state.auth_sub` /
`auth_team_id`, but get_current_user never wrote them — so the X-Titus-User-Id
header was always the authoritative metering identity (spoofable) and team
attribution was always None. get_current_user now writes both, activating the
existing resolve_caller_id Fix-12. The metering middleware also skips 4xx so
unauthenticated/failed requests cannot inflate the unique-user count.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from tidus.auth.middleware import get_current_user
from tidus.metering.middleware import MeteringMiddleware
from tidus.settings import Settings


def _request():
    return Request({"type": "http", "method": "GET", "path": "/api/v1/route", "headers": [], "state": {}})


def _settings(environment, oidc_issuer_url=""):
    return Settings(environment=environment, oidc_issuer_url=oidc_issuer_url)


# ── AUTH-2: dev-mode gate ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dev_mode_blocked_for_unknown_environment():
    """A non-production, non-dev environment string must NOT grant admin."""
    with pytest.raises(HTTPException) as exc:
        await get_current_user(_request(), None, _settings("staging"))
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_dev_mode_blocked_for_production():
    with pytest.raises(HTTPException) as exc:
        await get_current_user(_request(), None, _settings("production"))
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_dev_mode_allowed_for_development():
    user = await get_current_user(_request(), None, _settings("development"))
    assert user.role == "admin"


# ── ISO-9 / ISO-8: get_current_user wires identity to request.state ──────────────

@pytest.mark.asyncio
async def test_dev_mode_wires_auth_sub_and_team_to_request_state():
    req = _request()
    await get_current_user(req, None, _settings("development"))
    assert req.state.auth_sub == "dev"
    assert req.state.auth_team_id == "team-dev"


# ── ISO-9: metering middleware skips 4xx (no inflation from failed requests) ─────

def _metered_request(path="/api/v1/route"):
    return Request({
        "type": "http", "method": "POST", "path": path,
        "headers": [(b"x-titus-user-id", b"attacker-supplied")],
        "client": ("1.2.3.4", 1234), "state": {},
    })


@pytest.mark.asyncio
async def test_metering_skips_4xx_responses():
    from unittest.mock import AsyncMock, MagicMock
    service = MagicMock()
    service.record_event = AsyncMock()
    mw = MeteringMiddleware(MagicMock(), metering_getter=lambda: service)

    async def call_next(_req):
        return Response(status_code=401)

    resp = await mw.dispatch(_metered_request(), call_next)
    await asyncio.sleep(0)  # let any (erroneously) scheduled task run
    assert resp.status_code == 401
    service.record_event.assert_not_called()


@pytest.mark.asyncio
async def test_metering_records_2xx_responses():
    from unittest.mock import AsyncMock, MagicMock
    service = MagicMock()
    service.record_event = AsyncMock()
    mw = MeteringMiddleware(MagicMock(), metering_getter=lambda: service)

    async def call_next(_req):
        return Response(status_code=200)

    await mw.dispatch(_metered_request(), call_next)
    await asyncio.sleep(0)  # let the fire-and-forget metering task run
    service.record_event.assert_awaited_once()
