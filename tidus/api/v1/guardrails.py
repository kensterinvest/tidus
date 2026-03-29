"""POST/DELETE /api/v1/guardrails/sessions — Agent session management.

Endpoints:
    POST   /api/v1/guardrails/sessions          — Start a new agent session
    GET    /api/v1/guardrails/sessions/{id}     — Get session state
    DELETE /api/v1/guardrails/sessions/{id}     — Terminate session

Example curl:
    curl -X POST http://localhost:8000/api/v1/guardrails/sessions \\
      -H "Content-Type: application/json" \\
      -d '{"session_id":"sess-001","team_id":"team-eng"}'
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from tidus.api.deps import get_agent_guard, get_guardrail_policy, get_session_store
from tidus.auth.middleware import TokenPayload, get_current_user
from tidus.auth.rbac import Role, require_role
from tidus.guardrails.agent_guard import AgentGuard
from tidus.guardrails.session_store import SessionStore
from tidus.models.guardrails import AgentSession, GuardrailPolicy

router = APIRouter(prefix="/guardrails", tags=["Guardrails"])


# ── Request models ────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    session_id: str
    team_id: str | None = None


class AdvanceRequest(BaseModel):
    """Request to check and advance an agent session by one step."""
    session_id: str
    input_tokens: int


# ── Route handlers ────────────────────────────────────────────────────────────

@router.post(
    "/sessions",
    response_model=AgentSession,
    status_code=201,
    summary="Start a new agent session",
)
async def create_session(
    body: CreateSessionRequest,
    store: Annotated[SessionStore, Depends(get_session_store)],
    policy: Annotated[GuardrailPolicy, Depends(get_guardrail_policy)],
    _auth: Annotated[TokenPayload, Depends(require_role(
        Role.developer, Role.team_manager, Role.admin, Role.service_account,
    ))],
) -> AgentSession:
    """Create a new agent session with the current guardrail policy.

    Raises 409 if the session_id already exists.
    """
    try:
        return await store.create(body.session_id, policy, team_id=body.team_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/sessions/{session_id}",
    response_model=AgentSession,
    summary="Get current state of an agent session",
)
async def get_session(
    session_id: str,
    store: Annotated[SessionStore, Depends(get_session_store)],
    _auth: Annotated[TokenPayload, Depends(get_current_user)],
) -> AgentSession:
    session = await store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


@router.delete(
    "/sessions/{session_id}",
    status_code=204,
    summary="Terminate and remove an agent session",
)
async def terminate_session(
    session_id: str,
    store: Annotated[SessionStore, Depends(get_session_store)],
    _auth: Annotated[TokenPayload, Depends(require_role(
        Role.developer, Role.team_manager, Role.admin,
    ))],
) -> None:
    removed = await store.terminate(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.post(
    "/sessions/advance",
    summary="Check guardrails and advance session depth by one step",
)
async def advance_session(
    body: AdvanceRequest,
    guard: Annotated[AgentGuard, Depends(get_agent_guard)],
    _auth: Annotated[TokenPayload, Depends(require_role(
        Role.developer, Role.team_manager, Role.admin, Role.service_account,
    ))],
) -> dict:
    """Validate that this agent step is within policy limits and increment depth.

    Returns `{"allowed": true}` or `{"allowed": false, "reason": "..."}`.
    Use this before each step in a multi-step agent loop.
    """
    result = await guard.check_and_advance(body.session_id, body.input_tokens)
    return {"allowed": result.allowed, "reason": result.reason}
