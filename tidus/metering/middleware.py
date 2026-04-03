"""Metering middleware — intercepts routing requests to record AI user events.

Runs as a Starlette BaseHTTPMiddleware so it fires on every request without
requiring route-level dependency injection. Non-routing paths (health, docs,
metrics, dashboard static) are skipped so they don't inflate AI user counts.

The caller identity is resolved per TIT-32 spec:
  1. X-Titus-User-Id header
  2. Authenticated sub (stored on request.state by auth middleware)
  3. SHA-256(IP + User-Agent) anonymous fingerprint

`metering_getter` is a callable (() -> MeteringService) resolved lazily on
the first request. This avoids a circular import with the lifespan startup,
since the MeteringService singleton is built after app construction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from tidus.metering.service import MeteringService, resolve_caller_id

# Only count requests to these routing API paths
_METERED_PREFIXES = ("/api/v1/route", "/api/v1/complete")


class MeteringMiddleware(BaseHTTPMiddleware):
    """Records one AI user event per metered request (non-fatal, fire-and-forget)."""

    def __init__(self, app, metering_getter: Callable[[], MeteringService]) -> None:
        super().__init__(app)
        self._getter = metering_getter
        self._service: MeteringService | None = None

    def _get_service(self) -> MeteringService:
        if self._service is None:
            self._service = self._getter()
        return self._service

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        path = request.url.path
        if not any(path.startswith(p) for p in _METERED_PREFIXES):
            return response

        # Skip 5xx infra failures — only count meaningful routing attempts
        if response.status_code >= 500:
            return response

        user_id_header = request.headers.get("x-titus-user-id")
        api_key_sub = getattr(request.state, "auth_sub", None)
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        team_id = getattr(request.state, "auth_team_id", None)

        caller_id, caller_source = resolve_caller_id(
            user_id_header=user_id_header,
            api_key_sub=api_key_sub,
            client_ip=client_ip,
            user_agent=user_agent,
        )

        # Fire-and-forget — never block the response path
        task = asyncio.create_task(
            self._get_service().record_event(
                caller_id=caller_id,
                caller_source=caller_source,
                team_id=team_id,
                path=path,
            )
        )
        task.add_done_callback(
            lambda t: t.exception() and __import__("structlog").get_logger(__name__).warning(
                "metering_record_failed", error=str(t.exception())
            )
        )

        return response
