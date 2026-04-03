"""POST /api/v1/complete — route AND execute a task in one call.

This is the full AI proxy endpoint: Tidus selects the best model,
calls the vendor adapter, logs the cost, and deducts from the budget.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from tidus.adapters.adapter_factory import get_adapter
from tidus.api.deps import (
    get_audit_logger,
    get_cost_logger,
    get_enforcer,
    get_registry,
    get_selector,
)
from tidus.audit.logger import AuditLogger
from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role, require_role
from tidus.budget.enforcer import BudgetEnforcer
from tidus.cost.logger import CostLogger
from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor
from tidus.router.registry import ModelRegistry
from tidus.router.selector import ModelSelectionError, ModelSelector

log = structlog.get_logger(__name__)

router = APIRouter(tags=["complete"])


# ── Request / Response models ─────────────────────────────────────────────────

class CompleteRequest(BaseModel):
    team_id: str
    complexity: Complexity
    domain: Domain
    estimated_input_tokens: int = Field(ge=1)
    messages: list[dict]
    privacy: Privacy = Privacy.public
    estimated_output_tokens: int = Field(256, ge=1)
    agent_depth: int = Field(0, ge=0, le=5)
    preferred_model_id: Optional[str] = None
    max_cost_usd: Optional[float] = None
    workflow_id: Optional[str] = None
    agent_session_id: Optional[str] = None
    stream: bool = False


class CompleteResponse(BaseModel):
    task_id: str
    chosen_model_id: str
    vendor: str
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    finish_reason: str
    fallback_from: Optional[str] = None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/complete", response_model=CompleteResponse)
async def complete(
    request: Request,
    req: CompleteRequest,
    selector: Annotated[ModelSelector, Depends(get_selector)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    enforcer: Annotated[BudgetEnforcer, Depends(get_enforcer)],
    cost_logger: Annotated[CostLogger, Depends(get_cost_logger)],
    audit: Annotated[AuditLogger, Depends(get_audit_logger)],
    _auth: Annotated[TokenPayload, Depends(require_role(
        Role.developer, Role.team_manager, Role.admin, Role.service_account,
    ))],
) -> CompleteResponse:
    """Route and execute a task. Returns the model response with cost metadata."""
    # JWT team_id takes precedence over body — prevents cross-team budget abuse.
    effective_team_id = _auth.team_id or req.team_id

    task = TaskDescriptor(
        task_id=str(uuid.uuid4()),
        team_id=effective_team_id,
        workflow_id=req.workflow_id,
        agent_session_id=req.agent_session_id,
        agent_depth=req.agent_depth,
        complexity=req.complexity,
        domain=req.domain,
        privacy=req.privacy,
        estimated_input_tokens=req.estimated_input_tokens,
        estimated_output_tokens=req.estimated_output_tokens,
        messages=req.messages,
        preferred_model_id=req.preferred_model_id,
        max_cost_usd=req.max_cost_usd,
    )

    # Stage 1–5: select the model
    try:
        decision = await selector.select(task)
    except ModelSelectionError as exc:
        # Derive dominant rejection reason from the last stage's rejections
        failure_reason = (
            exc.rejections[-1].rejection_reason if exc.rejections else "unknown"
        )
        log.warning(
            "complete_no_model",
            task_id=task.task_id,
            stage=exc.stage,
            reason=failure_reason,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "detail": "No capable model found",
                "failure_stage": exc.stage,
                "failure_reason": failure_reason,
                "rejections": [
                    {"model_id": r.chosen_model_id, "reason": r.rejection_reason}
                    for r in exc.rejections
                ],
            },
        )

    spec = registry.get(decision.chosen_model_id)
    if spec is None:
        raise HTTPException(status_code=500, detail="Selected model not found in registry")

    # Get the adapter and execute
    try:
        adapter = get_adapter(spec.vendor)
    except KeyError:
        raise HTTPException(
            status_code=501,
            detail=f"No adapter available for vendor '{spec.vendor}'. "
                   "Check that the vendor adapter is installed and registered.",
        )

    try:
        response = await adapter.complete(decision.chosen_model_id, task)
    except Exception as exc:
        log.error(
            "adapter_error",
            task_id=task.task_id,
            model_id=decision.chosen_model_id,
            vendor=spec.vendor,
            error=str(exc),
        )
        # Try the first fallback if available
        if spec.fallbacks:
            fallback_id = spec.fallbacks[0]
            fallback_spec = registry.get(fallback_id)
            if fallback_spec:
                try:
                    fallback_adapter = get_adapter(fallback_spec.vendor)
                    response = await fallback_adapter.complete(fallback_id, task)
                    decision.fallback_from = decision.chosen_model_id
                    decision = decision.model_copy(
                        update={"chosen_model_id": fallback_id, "fallback_from": decision.chosen_model_id}
                    )
                    spec = fallback_spec
                    log.info(
                        "fallback_used",
                        task_id=task.task_id,
                        primary=decision.fallback_from,
                        fallback=fallback_id,
                    )
                except Exception as fallback_exc:
                    log.error("fallback_also_failed", error=str(fallback_exc))
                    raise HTTPException(
                        status_code=502,
                        detail="Upstream model unavailable and fallback also failed. "
                               "Check server logs for details.",
                    )
            else:
                raise HTTPException(
                    status_code=502,
                    detail="Upstream model unavailable. Check server logs for details.",
                )
        else:
            raise HTTPException(
                status_code=502,
                detail="Upstream model unavailable. Check server logs for details.",
            )

    # Deduct actual cost from budget
    actual_cost = (
        response.input_tokens / 1000 * spec.input_price
        + response.output_tokens / 1000 * spec.output_price
    )
    await enforcer.deduct(effective_team_id, req.workflow_id, actual_cost)

    # Log cost to DB (non-fatal) — pass actual_cost so DB reflects real billed amount
    await cost_logger.record(task, decision, response, spec.vendor, actual_cost)

    # Audit trail (non-fatal)
    await audit.record(
        actor=_auth,
        action="complete",
        resource_type="task",
        resource_id=task.task_id,
        outcome="success",
        metadata={
            "chosen_model_id": response.model_id,
            "vendor": spec.vendor,
            "cost_usd": round(actual_cost, 6),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "fallback_from": decision.fallback_from,
        },
    )

    log.info(
        "complete_success",
        task_id=task.task_id,
        model_id=response.model_id,
        cost_usd=round(actual_cost, 6),
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )

    return CompleteResponse(
        task_id=task.task_id,
        chosen_model_id=response.model_id,
        vendor=spec.vendor,
        content=response.content,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=actual_cost,
        latency_ms=round(response.latency_ms, 1),
        finish_reason=response.finish_reason,
        fallback_from=decision.fallback_from,
    )
