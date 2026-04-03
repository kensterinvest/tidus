"""GET /api/v1/reports/monthly — Monthly AI cost savings report.

Returns a self-contained savings report for a calendar month:
  - Actual spend (what Tidus actually cost)
  - Baseline spend (what it would have cost if everything went to a premium model)
  - Estimated savings in USD and percentage
  - Per-day breakdown for trend visibility
  - Top models by traffic share

All data is computed from the local database — nothing is sent to any external
service. The 'note' field in every response makes this explicit.

Example:
    curl "http://localhost:8000/api/v1/reports/monthly?year=2026&month=4" | jq .
    curl "http://localhost:8000/api/v1/reports/monthly" | jq .savings_pct
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from tidus.api.deps import get_enforcer, get_registry
from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role, require_role
from tidus.budget.enforcer import BudgetEnforcer
from tidus.router.registry import ModelRegistry

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/reports", tags=["Reports"])

# Default baseline: the most expensive common model — makes savings look realistic
_DEFAULT_BASELINE = "claude-opus-4-6"
# Fallback pricing if the baseline model is not in the registry (per 1K tokens)
_FALLBACK_INPUT_PRICE = 0.005
_FALLBACK_OUTPUT_PRICE = 0.025


# ── Response models ───────────────────────────────────────────────────────────

class TopModel(BaseModel):
    model_id: str
    vendor: str
    requests: int
    cost_usd: float
    pct_of_traffic: float


class DailyBreakdown(BaseModel):
    date: str       # "YYYY-MM-DD"
    requests: int
    cost_usd: float
    savings_usd: float


class MonthlySavingsReport(BaseModel):
    period: str                    # "2026-04"
    team_id: str                   # "all" or specific team
    total_requests: int
    total_cost_usd: float
    baseline_cost_usd: float       # if all requests went to baseline_model_id
    estimated_savings_usd: float
    savings_pct: float
    avg_cost_per_request_usd: float
    baseline_model_id: str
    top_models: list[TopModel]
    daily_breakdown: list[DailyBreakdown]
    generated_at: str
    note: str


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _fetch_records(year: int, month: int, team_id: str | None) -> list[dict]:
    """Fetch cost records from DB for the given calendar month."""
    try:
        from tidus.db.engine import CostRecordORM, get_session_factory
        from sqlalchemy import select

        start = datetime(year, month, 1, tzinfo=timezone.utc)
        # First day of next month
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        session_factory = get_session_factory()
        async with session_factory() as session:
            stmt = select(CostRecordORM).where(
                CostRecordORM.timestamp >= start,
                CostRecordORM.timestamp < end,
            )
            if team_id and team_id.lower() != "all":
                stmt = stmt.where(CostRecordORM.team_id == team_id)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "model_id": r.model_id,
                    "vendor": r.vendor,
                    "team_id": r.team_id,
                    "cost_usd": r.cost_usd or 0.0,
                    "input_tokens": r.input_tokens or 0,
                    "output_tokens": r.output_tokens or 0,
                    "latency_ms": r.latency_ms or 0.0,
                    "timestamp": r.timestamp,
                }
                for r in rows
            ]
    except Exception as exc:
        log.warning("reports_db_read_failed", error=str(exc))
        return []


def _baseline_prices(registry: ModelRegistry, baseline_model_id: str) -> tuple[float, float]:
    """Return (input_price_per_1k, output_price_per_1k) for the baseline model."""
    spec = registry.get(baseline_model_id)
    if spec:
        return spec.input_price, spec.output_price
    return _FALLBACK_INPUT_PRICE, _FALLBACK_OUTPUT_PRICE


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get(
    "/monthly",
    response_model=MonthlySavingsReport,
    summary="Monthly AI cost savings report (fully local, no external dependencies)",
)
async def monthly_savings_report(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    _auth: Annotated[TokenPayload, Depends(require_role(
        Role.read_only, Role.developer, Role.team_manager, Role.admin,
    ))],
    year: int = Query(default=None, ge=2024, le=2099, description="Report year (defaults to current)"),
    month: int = Query(default=None, ge=1, le=12, description="Report month 1–12 (defaults to current)"),
    team_id: str = Query(default="all", description="Team ID, or 'all' for all teams"),
    baseline_model_id: str = Query(
        default=_DEFAULT_BASELINE,
        description="Model to use as 'always-premium' baseline for savings calculation",
    ),
) -> MonthlySavingsReport:
    """Return a monthly savings report computed entirely from local data.

    Compares actual Tidus spend against what the same volume of requests would
    have cost if every one was routed to a premium baseline model.

    Non-admin callers are restricted to their own team's data.

    The report is purely local — no data is sent to any external service.
    """
    now = datetime.now(timezone.utc)
    report_year = year if year is not None else now.year
    report_month = month if month is not None else now.month

    # Non-admin callers can only see their own team
    effective_team = team_id
    if _auth.role not in (Role.admin, Role.team_manager) and team_id.lower() == "all":
        effective_team = _auth.team_id

    records = await _fetch_records(report_year, report_month, effective_team)

    total_requests = len(records)
    total_cost = sum(r["cost_usd"] for r in records)
    total_input_tokens = sum(r["input_tokens"] for r in records)
    total_output_tokens = sum(r["output_tokens"] for r in records)

    # Baseline cost = what the same token volume would cost at premium pricing
    in_price, out_price = _baseline_prices(registry, baseline_model_id)
    baseline_cost = (
        total_input_tokens / 1000 * in_price
        + total_output_tokens / 1000 * out_price
    )
    savings = max(0.0, baseline_cost - total_cost)
    savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

    # Per-model aggregation
    model_stats: dict[str, dict] = defaultdict(lambda: {
        "vendor": "", "requests": 0, "cost": 0.0,
    })
    for r in records:
        mid = r["model_id"]
        model_stats[mid]["vendor"] = r["vendor"]
        model_stats[mid]["requests"] += 1
        model_stats[mid]["cost"] += r["cost_usd"]

    top_models = sorted(
        [
            TopModel(
                model_id=mid,
                vendor=stats["vendor"],
                requests=stats["requests"],
                cost_usd=round(stats["cost"], 6),
                pct_of_traffic=round(stats["requests"] / max(total_requests, 1) * 100, 1),
            )
            for mid, stats in model_stats.items()
        ],
        key=lambda m: m.requests,
        reverse=True,
    )[:10]

    # Daily breakdown
    daily: dict[str, dict] = defaultdict(lambda: {
        "requests": 0, "cost": 0.0,
        "input_tokens": 0, "output_tokens": 0,
    })
    for r in records:
        ts = r["timestamp"]
        if ts is None:
            continue
        if hasattr(ts, "date"):
            day = ts.date().isoformat()
        else:
            day = str(ts)[:10]
        daily[day]["requests"] += 1
        daily[day]["cost"] += r["cost_usd"]
        daily[day]["input_tokens"] += r["input_tokens"]
        daily[day]["output_tokens"] += r["output_tokens"]

    daily_breakdown = sorted(
        [
            DailyBreakdown(
                date=day,
                requests=stats["requests"],
                cost_usd=round(stats["cost"], 6),
                savings_usd=round(
                    max(0.0,
                        stats["input_tokens"] / 1000 * in_price
                        + stats["output_tokens"] / 1000 * out_price
                        - stats["cost"]
                    ),
                    6,
                ),
            )
            for day, stats in daily.items()
        ],
        key=lambda d: d.date,
    )

    period_label = f"{report_year}-{report_month:02d}"

    log.info(
        "monthly_report_generated",
        period=period_label,
        team_id=effective_team,
        total_requests=total_requests,
        savings_pct=round(savings_pct, 1),
    )

    return MonthlySavingsReport(
        period=period_label,
        team_id=effective_team,
        total_requests=total_requests,
        total_cost_usd=round(total_cost, 6),
        baseline_cost_usd=round(baseline_cost, 6),
        estimated_savings_usd=round(savings, 6),
        savings_pct=round(savings_pct, 2),
        avg_cost_per_request_usd=round(total_cost / max(total_requests, 1), 6),
        baseline_model_id=baseline_model_id,
        top_models=top_models,
        daily_breakdown=daily_breakdown,
        generated_at=now.isoformat(),
        note=(
            "All data is computed from your local Tidus database. "
            "Nothing is sent to any external service."
        ),
    )
