"""POST /api/v1/complete — route AND execute a task in one call.

This is the full AI proxy endpoint: Tidus selects the best model,
calls the vendor adapter, logs the cost, and deducts from the budget.
"""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from tidus.adapters.adapter_factory import get_adapter
from tidus.api.deps import (
    get_agent_guard,
    get_audit_logger,
    get_cost_logger,
    get_enforcer,
    get_exact_cache,
    get_registry,
    get_selector,
    get_session_store,
)
from tidus.audit.logger import AuditLogger
from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role, require_role
from tidus.budget.enforcer import BudgetEnforcer
from tidus.cache.exact_cache import ExactCache
from tidus.cost.logger import CostLogger
from tidus.guardrails.agent_guard import AgentGuard
from tidus.guardrails.session_store import SessionStore
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
    preferred_model_id: str | None = None
    max_cost_usd: float | None = None
    workflow_id: str | None = None
    agent_session_id: str | None = None
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
    fallback_from: str | None = None
    cache_hit: bool = False


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
    session_store: Annotated[SessionStore, Depends(get_session_store)],
    agent_guard: Annotated[AgentGuard, Depends(get_agent_guard)],
    exact_cache: Annotated[ExactCache | None, Depends(get_exact_cache)],
    _auth: Annotated[TokenPayload, Depends(require_role(
        Role.developer, Role.team_manager, Role.admin, Role.service_account,
    ))],
) -> CompleteResponse:
    """Route and execute a task. Returns the model response with cost metadata."""
    # JWT team_id takes precedence over body — prevents cross-team budget abuse.
    effective_team_id = _auth.team_id or req.team_id

    # Assign task_id up front so every error path can audit against the same ID.
    task_id = str(uuid.uuid4())

    async def _audit_error(status_code: int, reason: str, **extra) -> None:
        """Record an audit entry for a failed request (Fix 20). Non-fatal."""
        try:
            await audit.record(
                actor=_auth,
                action="complete",
                resource_type="task",
                resource_id=task_id,
                outcome="error",
                metadata={"reason": reason, "http_status": status_code, **extra},
            )
        except Exception as exc:
            log.warning("audit_error_record_failed", task_id=task_id, error=str(exc))

    # ── Agent-depth gate (Fix 3: Option a — hard break) ──────────────────────
    # agent_depth > 0 requires a server-tracked session. The server — not the
    # client — is the source of truth for the current depth; AgentGuard advances
    # the session and rejects requests that exceed max_agent_depth.
    server_depth = 0
    if req.agent_depth > 0:
        if not req.agent_session_id:
            await _audit_error(400, "agent_depth_without_session")
            raise HTTPException(
                status_code=400,
                detail="agent_depth > 0 requires an agent_session_id created via "
                       "POST /api/v1/guardrails/sessions.",
            )
        session = await session_store.get(req.agent_session_id)
        if session is None:
            await _audit_error(400, "unknown_agent_session", session_id=req.agent_session_id)
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agent session '{req.agent_session_id}'.",
            )
        # Cross-team session use is not allowed.
        is_cross_team = _auth.role in {Role.admin.value, Role.team_manager.value}
        if session.team_id and session.team_id != effective_team_id and not is_cross_team:
            await _audit_error(
                400, "cross_team_session_access",
                session_id=req.agent_session_id,
                session_team=session.team_id,
            )
            raise HTTPException(
                status_code=400,
                detail=f"Agent session '{req.agent_session_id}' not owned by caller's team.",
            )
        guard_result = await agent_guard.check_and_advance(
            req.agent_session_id, req.estimated_input_tokens,
        )
        if not guard_result.allowed:
            await _audit_error(
                429, "agent_guardrail_exceeded",
                session_id=req.agent_session_id,
                guard_reason=guard_result.reason,
            )
            raise HTTPException(
                status_code=429,
                detail=guard_result.reason or "Agent guardrail exceeded",
            )
        server_depth = guard_result.session.current_depth if guard_result.session else 0

    task = TaskDescriptor(
        task_id=task_id,
        team_id=effective_team_id,
        workflow_id=req.workflow_id,
        agent_session_id=req.agent_session_id,
        # Authoritative depth from the server session, NOT the client body.
        agent_depth=server_depth,
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
        await _audit_error(
            422, "no_capable_model",
            failure_stage=exc.stage,
            failure_reason=str(failure_reason),
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
        await _audit_error(
            500, "selected_model_not_in_registry",
            chosen_model_id=decision.chosen_model_id,
        )
        raise HTTPException(status_code=500, detail="Selected model not found in registry")

    # ── ExactCache lookup (Fix 11) ───────────────────────────────────────────
    # Confidential tasks bypass the cache entirely — no content from sensitive
    # prompts is ever persisted outside the one-shot adapter call.
    cache_key: str | None = None
    cache_eligible = exact_cache is not None and req.privacy != Privacy.confidential
    if cache_eligible:
        cache_key = exact_cache.make_key(
            effective_team_id, req.messages, decision.chosen_model_id,
        )
        cached_content = await exact_cache.get(cache_key)
        if cached_content is not None:
            log.info(
                "complete_cache_hit",
                task_id=task.task_id,
                model_id=decision.chosen_model_id,
            )
            return CompleteResponse(
                task_id=task.task_id,
                chosen_model_id=decision.chosen_model_id,
                vendor=spec.vendor,
                content=cached_content,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=0.0,
                finish_reason="stop",
                fallback_from=None,
                cache_hit=True,
            )

    # Get the adapter and execute
    try:
        adapter = get_adapter(spec.vendor)
    except KeyError:
        await _audit_error(
            501, "no_adapter_for_vendor",
            vendor=spec.vendor, chosen_model_id=decision.chosen_model_id,
        )
        raise HTTPException(
            status_code=501,
            detail=f"No adapter available for vendor '{spec.vendor}'. "
                   "Check that the vendor adapter is installed and registered.",
        )

    # Atomic reservation — prevents concurrent requests from collectively
    # overrunning the hard-stop limit between Stage 4 check and deduct().
    estimated_cost = decision.estimated_cost_usd or 0.0
    if estimated_cost > 0 and not await enforcer.reserve(
        effective_team_id, req.workflow_id, estimated_cost,
    ):
        log.warning(
            "budget_reservation_failed",
            task_id=task.task_id,
            team_id=effective_team_id,
            workflow_id=req.workflow_id,
            estimated_cost=estimated_cost,
        )
        await _audit_error(
            402, "budget_reservation_failed",
            chosen_model_id=decision.chosen_model_id,
            estimated_cost=estimated_cost,
        )
        raise HTTPException(
            status_code=402,
            detail="Budget would be exceeded by concurrent requests — try again later.",
        )

    try:
        response = await adapter.complete(decision.chosen_model_id, task)
    except Exception as exc:
        if estimated_cost > 0:
            await enforcer.refund(effective_team_id, req.workflow_id, estimated_cost)
        log.error(
            "adapter_error",
            task_id=task.task_id,
            model_id=decision.chosen_model_id,
            vendor=spec.vendor,
            error=str(exc),
        )

        # Re-run the full 5-stage selector excluding the failed model. This
        # preserves privacy routing, tier ceilings, budget, and guardrails on
        # the fallback — unlike the old code which short-circuited to
        # `spec.fallbacks[0]` and skipped every stage.
        primary_model_id = decision.chosen_model_id
        try:
            decision = await selector.select(
                task, exclude_model_ids=frozenset({primary_model_id}),
            )
        except ModelSelectionError:
            await _audit_error(
                502, "no_eligible_fallback", primary_model_id=primary_model_id,
            )
            raise HTTPException(
                status_code=502,
                detail="Upstream model unavailable and no eligible fallback. "
                       "Check server logs for details.",
            )

        spec = registry.get(decision.chosen_model_id)
        if spec is None:
            await _audit_error(
                500, "fallback_model_missing_from_registry",
                fallback_model_id=decision.chosen_model_id,
            )
            raise HTTPException(
                status_code=500,
                detail="Fallback model disappeared from registry mid-request",
            )
        try:
            fallback_adapter = get_adapter(spec.vendor)
        except KeyError:
            await _audit_error(
                502, "no_adapter_for_fallback_vendor",
                vendor=spec.vendor, fallback_model_id=decision.chosen_model_id,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Fallback vendor '{spec.vendor}' has no adapter installed.",
            )

        # Reservation for the fallback cost (new estimated)
        estimated_cost = decision.estimated_cost_usd or 0.0
        if estimated_cost > 0 and not await enforcer.reserve(
            effective_team_id, req.workflow_id, estimated_cost,
        ):
            await _audit_error(
                402, "fallback_budget_reservation_failed",
                fallback_model_id=decision.chosen_model_id,
                estimated_cost=estimated_cost,
            )
            raise HTTPException(
                status_code=402,
                detail="Budget would be exceeded by fallback — try again later.",
            )

        try:
            response = await fallback_adapter.complete(decision.chosen_model_id, task)
        except Exception as fallback_exc:
            if estimated_cost > 0:
                await enforcer.refund(effective_team_id, req.workflow_id, estimated_cost)
            log.error(
                "fallback_also_failed",
                primary=primary_model_id,
                fallback=decision.chosen_model_id,
                error=str(fallback_exc),
            )
            await _audit_error(
                502, "fallback_also_failed",
                primary_model_id=primary_model_id,
                fallback_model_id=decision.chosen_model_id,
            )
            raise HTTPException(
                status_code=502,
                detail="Upstream model unavailable and fallback also failed. "
                       "Check server logs for details.",
            )

        decision = decision.model_copy(update={"fallback_from": primary_model_id})
        log.info(
            "fallback_used",
            task_id=task.task_id,
            primary=primary_model_id,
            fallback=decision.chosen_model_id,
        )

    # Deduct actual cost from budget — adjusts the prior reservation by the
    # difference (actual - estimated) so the counter ends at actual cost.
    actual_cost = (
        response.input_tokens / 1000 * spec.input_price
        + response.output_tokens / 1000 * spec.output_price
    )
    reserved = estimated_cost if estimated_cost > 0 else None
    await enforcer.deduct(
        effective_team_id, req.workflow_id, actual_cost, reserved_usd=reserved,
    )

    # Log cost to DB (non-fatal) — pass actual_cost so DB reflects real billed amount
    await cost_logger.record(task, decision, response, spec.vendor, actual_cost)

    # Populate ExactCache (skip for confidential — guarded above)
    if cache_key is not None and exact_cache is not None:
        await exact_cache.set(cache_key, response.content, response.model_id)

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
