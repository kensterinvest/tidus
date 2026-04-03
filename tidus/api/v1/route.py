"""POST /api/v1/route — Route-only endpoint (no model execution).

Returns a RoutingDecision for a described task without calling any vendor API.
Use this for dry-run routing, cost estimation, and debugging selection logic.

Example curl:
    curl -X POST http://localhost:8000/api/v1/route \\
      -H "Content-Type: application/json" \\
      -d '{"team_id":"team-eng","complexity":"simple","domain":"chat",
           "estimated_input_tokens":500,"messages":[{"role":"user","content":"hi"}]}'
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from tidus.api.deps import get_audit_logger, get_selector
from tidus.audit.logger import AuditLogger
from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role, require_role
from tidus.models.routing import RejectionReason, RoutingDecision
from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor
from tidus.router.selector import ModelSelectionError, ModelSelector

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/route", tags=["Routing"])


# ── Request / Response models ─────────────────────────────────────────────────

class RouteRequest(BaseModel):
    """Routing request — describes the task to be routed.

    All fields except team_id, complexity, domain, and messages have
    sensible defaults to minimise required fields for simple requests.
    """
    team_id: str = Field(..., description="Team making the request")
    complexity: Complexity = Field(..., description="Task complexity tier")
    domain: Domain = Field(..., description="Primary task domain")
    privacy: Privacy = Field(Privacy.public, description="Data sensitivity level")
    estimated_input_tokens: int = Field(500, gt=0, description="Estimated input tokens")
    estimated_output_tokens: int = Field(256, gt=0, description="Estimated output tokens")
    messages: list[dict] = Field(
        default_factory=lambda: [{"role": "user", "content": ""}],
        description="OpenAI-compatible message list",
    )
    preferred_model_id: str | None = Field(None, description="Pin to a specific model if eligible")
    max_cost_usd: float | None = Field(None, gt=0, description="Hard per-request cost ceiling")
    agent_depth: int = Field(0, ge=0, description="Current agent recursion depth")
    workflow_id: str | None = None
    agent_session_id: str | None = None


class RejectionDetail(BaseModel):
    model_id: str
    reason: RejectionReason


class RouteResponse(BaseModel):
    """Routing decision returned to the caller."""
    task_id: str
    accepted: bool
    chosen_model_id: str | None = None
    estimated_cost_usd: float | None = None
    score: float | None = None
    # Populated only when accepted=False
    failure_stage: int | None = None
    rejections: list[RejectionDetail] | None = None


# ── Route handler ─────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=RouteResponse,
    summary="Route a task (no execution)",
    response_description="The routing decision for the described task",
)
async def route_task(
    request: Request,
    body: RouteRequest,
    selector: Annotated[ModelSelector, Depends(get_selector)],
    audit: Annotated[AuditLogger, Depends(get_audit_logger)],
    _auth: Annotated[TokenPayload, Depends(require_role(
        Role.developer, Role.team_manager, Role.admin, Role.service_account,
    ))],
) -> RouteResponse:
    """Run the 5-stage model selection algorithm and return the routing decision.

    Does **not** call any vendor API — use `POST /api/v1/complete` to execute.

    Raises:
        422: No model survived all 5 selection stages (budget, tier, guardrails, etc.)
    """
    task = TaskDescriptor(
        team_id=_auth.team_id or body.team_id,
        workflow_id=body.workflow_id,
        agent_session_id=body.agent_session_id,
        complexity=body.complexity,
        domain=body.domain,
        privacy=body.privacy,
        estimated_input_tokens=body.estimated_input_tokens,
        estimated_output_tokens=body.estimated_output_tokens,
        messages=body.messages,
        preferred_model_id=body.preferred_model_id,
        max_cost_usd=body.max_cost_usd,
        agent_depth=body.agent_depth,
    )

    try:
        decision: RoutingDecision = await selector.select(task)
    except ModelSelectionError as exc:
        log.warning(
            "routing_failed",
            team_id=body.team_id,
            stage=exc.stage,
            rejection_count=len(exc.rejections),
        )
        await audit.record(
            actor=_auth,
            action="route",
            resource_type="task",
            resource_id=task.task_id,
            outcome="rejected",
            rejection_reason=str(exc),
            metadata={"stage": exc.stage},
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "no_model_available",
                "message": str(exc),
                "failure_stage": exc.stage,
                "rejections": [
                    {"model_id": r.chosen_model_id, "reason": r.rejection_reason}
                    for r in exc.rejections
                    if r.rejection_reason is not None
                ],
            },
        ) from exc

    await audit.record(
        actor=_auth,
        action="route",
        resource_type="task",
        resource_id=decision.task_id,
        outcome="success",
        metadata={"chosen_model_id": decision.chosen_model_id, "estimated_cost_usd": decision.estimated_cost_usd},
    )

    return RouteResponse(
        task_id=decision.task_id,
        accepted=decision.accepted,
        chosen_model_id=decision.chosen_model_id,
        estimated_cost_usd=decision.estimated_cost_usd,
        score=decision.score,
    )
