"""Billing reconciliation API router.

Prefix: /api/v1/billing (registered in main.py)

Endpoints:
  POST /reconcile               — upload normalized billing CSV + trigger reconciliation
  GET  /reconciliations         — list reconciliation results (team-scoped for managers)
  GET  /reconciliations/summary — aggregate: total variance, counts by status
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import func, select

from tidus.api.deps import get_audit_logger, get_session_factory
from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role, require_role
from tidus.billing.csv_parser import BillingParseError, parse as parse_csv
from tidus.billing.reconciler import BillingReconciler
from tidus.db.registry_orm import BillingReconciliationORM

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/billing", tags=["Billing"])

_MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB hard cap for billing CSV uploads
_VALID_STATUSES = {"matched", "warning", "critical"}


# ── Response models ───────────────────────────────────────────────────────────

class ReconcileResponse(BaseModel):
    reconciliation_count: int
    matched: int
    warnings: int
    criticals: int
    total_variance_usd: float


class ReconciliationRow(BaseModel):
    id: str
    reconciliation_date: date
    model_id: str
    tidus_cost_usd: float
    provider_cost_usd: float
    variance_usd: float
    variance_pct: float
    status: str
    notes: str | None


class ReconciliationSummaryResponse(BaseModel):
    total_rows: int
    matched: int
    warnings: int
    criticals: int
    total_variance_usd: float
    critical_models: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/reconcile",
    response_model=ReconcileResponse,
    summary="Upload billing CSV and reconcile against Tidus cost records",
)
async def reconcile_billing(
    actor: Annotated[TokenPayload, Depends(require_role(Role.team_manager, Role.admin))],
    file: UploadFile = File(..., description="Normalized billing CSV (≤5 MB)"),
    date_from: date = Form(..., description="Inclusive start date (YYYY-MM-DD)"),
    date_to: date = Form(..., description="Inclusive end date (YYYY-MM-DD)"),
    replace_existing: bool = Form(
        default=False,
        description="If true, delete existing rows for the same (team, date range) before inserting. "
                    "Default false returns HTTP 409 if rows already exist.",
    ),
    session_factory=Depends(get_session_factory),
    audit_logger=Depends(get_audit_logger),
):
    if date_from > date_to:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="date_from must be ≤ date_to",
        )

    content = await file.read()

    # Reject uploads over the size limit before parsing
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Billing CSV must be ≤ {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    try:
        rows = parse_csv(content)
    except BillingParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    team_id = actor.team_id

    # Duplicate guard: check if rows already exist for this team + date range
    async with session_factory() as session:
        existing_count_result = await session.execute(
            select(func.count(BillingReconciliationORM.id)).where(
                BillingReconciliationORM.team_id == team_id,
                BillingReconciliationORM.reconciliation_date >= date_from,
                BillingReconciliationORM.reconciliation_date <= date_to,
            )
        )
        existing_count = existing_count_result.scalar_one()

    if existing_count > 0:
        if not replace_existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"{existing_count} reconciliation rows already exist for team "
                    f"'{team_id}' in range {date_from}–{date_to}. "
                    "Set replace_existing=true to overwrite."
                ),
            )
        # replace_existing=True: delete existing rows for this team + date range
        from sqlalchemy import delete as sa_delete
        async with session_factory() as session:
            await session.execute(
                sa_delete(BillingReconciliationORM).where(
                    BillingReconciliationORM.team_id == team_id,
                    BillingReconciliationORM.reconciliation_date >= date_from,
                    BillingReconciliationORM.reconciliation_date <= date_to,
                )
            )
            await session.commit()

    summary = await BillingReconciler().reconcile(
        rows=rows,
        date_from=date_from,
        date_to=date_to,
        session_factory=session_factory,
        uploaded_by=actor.sub,
        team_id=team_id,
        audit_logger=audit_logger,
    )

    return ReconcileResponse(
        reconciliation_count=summary.reconciliation_count,
        matched=summary.matched,
        warnings=summary.warnings,
        criticals=summary.criticals,
        total_variance_usd=summary.total_variance_usd,
    )


@router.get(
    "/reconciliations",
    response_model=list[ReconciliationRow],
    summary="List reconciliation results",
)
async def list_reconciliations(
    actor: Annotated[TokenPayload, Depends(require_role(Role.team_manager, Role.admin))],
    session_factory=Depends(get_session_factory),
    model_id: str | None = None,
    status_filter: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = 100,
    offset: int = 0,
):
    if status_filter is not None and status_filter not in _VALID_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"status_filter must be one of {sorted(_VALID_STATUSES)}",
        )
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="limit must be between 1 and 1000",
        )

    async with session_factory() as session:
        q = select(BillingReconciliationORM)

        # team_manager sees only their own team's reconciliations
        if actor.role != Role.admin.value:
            q = q.where(BillingReconciliationORM.team_id == actor.team_id)

        if model_id:
            q = q.where(BillingReconciliationORM.model_id == model_id)
        if status_filter:
            q = q.where(BillingReconciliationORM.status == status_filter)
        if date_from:
            q = q.where(BillingReconciliationORM.reconciliation_date >= date_from)
        if date_to:
            q = q.where(BillingReconciliationORM.reconciliation_date <= date_to)

        q = q.order_by(
            BillingReconciliationORM.reconciliation_date.desc(),
            BillingReconciliationORM.model_id,
        ).limit(limit).offset(offset)

        result = await session.execute(q)
        rows = result.scalars().all()

    return [
        ReconciliationRow(
            id=r.id,
            reconciliation_date=r.reconciliation_date,
            model_id=r.model_id,
            tidus_cost_usd=r.tidus_cost_usd,
            provider_cost_usd=r.provider_cost_usd,
            variance_usd=r.variance_usd,
            variance_pct=r.variance_pct,
            status=r.status,
            notes=r.notes,
        )
        for r in rows
    ]


@router.get(
    "/reconciliations/summary",
    response_model=ReconciliationSummaryResponse,
    summary="Aggregate summary of reconciliation results",
)
async def get_reconciliation_summary(
    actor: Annotated[TokenPayload, Depends(require_role(Role.team_manager, Role.admin))],
    session_factory=Depends(get_session_factory),
    date_from: date | None = None,
    date_to: date | None = None,
):
    """Return aggregated counts and total variance using SQL aggregation.

    Runs GROUP BY in the database — does not load individual rows into memory.
    """
    async with session_factory() as session:
        # Aggregate counts and total variance by status in a single query
        q = (
            select(
                BillingReconciliationORM.status,
                func.count(BillingReconciliationORM.id).label("cnt"),
                func.sum(BillingReconciliationORM.variance_usd).label("variance_sum"),
            )
            .group_by(BillingReconciliationORM.status)
        )

        if actor.role != Role.admin.value:
            q = q.where(BillingReconciliationORM.team_id == actor.team_id)
        if date_from:
            q = q.where(BillingReconciliationORM.reconciliation_date >= date_from)
        if date_to:
            q = q.where(BillingReconciliationORM.reconciliation_date <= date_to)

        agg_result = await session.execute(q)
        agg_rows = agg_result.all()

        # Fetch critical model IDs separately (bounded: only critical rows)
        crit_q = select(BillingReconciliationORM.model_id).where(
            BillingReconciliationORM.status == "critical"
        )
        if actor.role != Role.admin.value:
            crit_q = crit_q.where(BillingReconciliationORM.team_id == actor.team_id)
        if date_from:
            crit_q = crit_q.where(BillingReconciliationORM.reconciliation_date >= date_from)
        if date_to:
            crit_q = crit_q.where(BillingReconciliationORM.reconciliation_date <= date_to)
        crit_result = await session.execute(crit_q)
        critical_model_ids = sorted({r.model_id for r in crit_result.all()})

    if not agg_rows:
        return ReconciliationSummaryResponse(
            total_rows=0, matched=0, warnings=0, criticals=0,
            total_variance_usd=0.0, critical_models=[],
        )

    counts: dict[str, int] = {}
    variance_by_status: dict[str, float] = {}
    for row in agg_rows:
        counts[row.status] = row.cnt
        variance_by_status[row.status] = float(row.variance_sum or 0.0)

    total_rows = sum(counts.values())
    total_variance = sum(variance_by_status.values())

    return ReconciliationSummaryResponse(
        total_rows=total_rows,
        matched=counts.get("matched", 0),
        warnings=counts.get("warning", 0),
        criticals=counts.get("critical", 0),
        total_variance_usd=round(total_variance, 6),
        critical_models=critical_model_ids,
    )
