"""Dashboard API — aggregated metrics for the /dashboard SPA.

Returns pre-aggregated data so the frontend needs only one request per refresh.

GET /api/v1/dashboard/summary — total cost, request count, model distribution
GET /api/v1/dashboard/cost-by-model — 7-day cost breakdown per model
GET /api/v1/dashboard/budgets — all team budget utilization
GET /api/v1/dashboard/sessions — active agent sessions
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from tidus.api.deps import get_enforcer, get_registry, get_session_store
from tidus.budget.enforcer import BudgetEnforcer
from tidus.guardrails.session_store import SessionStore
from tidus.router.registry import ModelRegistry

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


# ── Response models ────────────────────────────────────────────────────────────

class CostSummary(BaseModel):
    total_cost_usd: float
    total_requests: int
    requests_today: int
    avg_cost_per_request_usd: float
    cheapest_model_used: str | None
    most_used_model: str | None


class ModelCostRow(BaseModel):
    model_id: str
    vendor: str
    tier: int
    requests: int
    total_cost_usd: float
    avg_latency_ms: float
    enabled: bool


class BudgetRow(BaseModel):
    team_id: str
    spent_usd: float
    limit_usd: float | None
    utilisation_pct: float | None
    is_hard_stopped: bool
    warn_threshold_pct: float | None


class SessionRow(BaseModel):
    session_id: str
    team_id: str
    agent_depth: int
    step_count: int
    total_tokens: int
    started_at: str


class DashboardSummary(BaseModel):
    cost: CostSummary
    cost_by_model: list[ModelCostRow]
    budgets: list[BudgetRow]
    sessions: list[SessionRow]
    registry_health: dict  # {model_id: enabled}
    generated_at: str


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _get_cost_records_7d() -> list[dict]:
    """Fetch cost records from DB for the past 7 days."""
    try:
        from tidus.db.engine import CostRecordORM, get_session_factory

        session_factory = get_session_factory()
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        async with session_factory() as session:
            from sqlalchemy import select
            stmt = select(CostRecordORM).where(CostRecordORM.timestamp >= cutoff)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "model_id": r.model_id,
                    "vendor": r.vendor,
                    "cost_usd": r.cost_usd,
                    "latency_ms": r.latency_ms,
                    "timestamp": r.timestamp,
                }
                for r in rows
            ]
    except Exception as exc:
        log.warning("dashboard_db_read_failed", error=str(exc))
        return []


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/summary", response_model=DashboardSummary, summary="All dashboard data in one call")
async def dashboard_summary(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    enforcer: Annotated[BudgetEnforcer, Depends(get_enforcer)],
    session_store: Annotated[SessionStore, Depends(get_session_store)],
) -> DashboardSummary:
    """Return all metrics needed to render the dashboard in one request."""
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # ── Cost records from DB ───────────────────────────────────────────────────
    records = await _get_cost_records_7d()

    total_cost = sum(r["cost_usd"] for r in records)
    total_requests = len(records)
    requests_today = sum(
        1 for r in records
        if r["timestamp"] and r["timestamp"].replace(tzinfo=timezone.utc) >= today_start
    )

    # Per-model aggregation
    model_stats: dict[str, dict] = defaultdict(lambda: {
        "requests": 0, "cost": 0.0, "latency_sum": 0.0,
    })
    for r in records:
        mid = r["model_id"]
        model_stats[mid]["requests"] += 1
        model_stats[mid]["cost"] += r["cost_usd"]
        model_stats[mid]["latency_sum"] += r["latency_ms"] or 0.0

    most_used = max(model_stats, key=lambda m: model_stats[m]["requests"]) if model_stats else None
    cheapest = min(
        (m for m in model_stats if model_stats[m]["cost"] > 0),
        key=lambda m: model_stats[m]["cost"] / max(model_stats[m]["requests"], 1),
        default=None,
    )

    cost_summary = CostSummary(
        total_cost_usd=round(total_cost, 6),
        total_requests=total_requests,
        requests_today=requests_today,
        avg_cost_per_request_usd=round(total_cost / max(total_requests, 1), 6),
        cheapest_model_used=cheapest,
        most_used_model=most_used,
    )

    # ── Cost by model rows ─────────────────────────────────────────────────────
    cost_by_model: list[ModelCostRow] = []
    for spec in registry.list_all():
        stats = model_stats.get(spec.model_id, {"requests": 0, "cost": 0.0, "latency_sum": 0.0})
        reqs = stats["requests"]
        cost_by_model.append(ModelCostRow(
            model_id=spec.model_id,
            vendor=spec.vendor,
            tier=spec.tier.value if hasattr(spec.tier, "value") else int(spec.tier),
            requests=reqs,
            total_cost_usd=round(stats["cost"], 6),
            avg_latency_ms=round(stats["latency_sum"] / max(reqs, 1), 1),
            enabled=spec.enabled,
        ))
    cost_by_model.sort(key=lambda r: r.total_cost_usd, reverse=True)

    # ── Budget rows ────────────────────────────────────────────────────────────
    budget_rows: list[BudgetRow] = []
    for policy in enforcer.list_policies():
        status = await enforcer.status(team_id=policy.scope_id)
        if status is None:
            continue
        budget_rows.append(BudgetRow(
            team_id=policy.scope_id,
            spent_usd=round(status.spent_usd, 6),
            limit_usd=status.limit_usd,
            utilisation_pct=status.utilisation_pct,
            is_hard_stopped=status.is_hard_stopped,
            warn_threshold_pct=getattr(policy, "warn_at_pct", None),
        ))

    # ── Active sessions ────────────────────────────────────────────────────────
    session_rows: list[SessionRow] = []
    for sess in await session_store.list_active():
        session_rows.append(SessionRow(
            session_id=sess.session_id,
            team_id=sess.team_id,
            agent_depth=sess.current_depth,
            step_count=sess.retry_count,
            total_tokens=sess.total_tokens_used,
            started_at=sess.started_at.isoformat(),
        ))

    # ── Registry health snapshot ───────────────────────────────────────────────
    registry_health = {
        spec.model_id: spec.enabled
        for spec in registry.list_all()
    }

    return DashboardSummary(
        cost=cost_summary,
        cost_by_model=cost_by_model,
        budgets=budget_rows,
        sessions=session_rows,
        registry_health=registry_health,
        generated_at=now.isoformat(),
    )
