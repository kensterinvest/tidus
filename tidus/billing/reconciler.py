"""BillingReconciler — compares Tidus cost_records against provider invoices.

For each (model_id, date) in the uploaded CSV, the reconciler:
  1. Queries cost_records to compute Tidus's internal cost for that model on that date.
  2. Computes variance: variance_usd = provider_cost - tidus_cost
                        variance_pct = variance_usd / provider_cost   (or 1.0 if provider=0)
  3. Assigns status:
       matched:  |variance_pct| ≤ 0.05
       warning:  0.05 < |variance_pct| ≤ 0.25
       critical: |variance_pct| > 0.25

All rows are written to billing_reconciliations regardless of status.
An audit entry is written with action='billing.reconcile'.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, UTC, timedelta

import structlog
from sqlalchemy import select

from tidus.billing.csv_parser import BillingRow
from tidus.db.engine import CostRecordORM
from tidus.db.registry_orm import BillingReconciliationORM

log = structlog.get_logger(__name__)

# Status thresholds
_MATCHED_MAX = 0.05
_WARNING_MAX = 0.25


def _classify_status(variance_pct: float) -> str:
    abs_pct = abs(variance_pct)
    if abs_pct <= _MATCHED_MAX:
        return "matched"
    if abs_pct <= _WARNING_MAX:
        return "warning"
    return "critical"


@dataclass
class ReconciliationSummary:
    reconciliation_count: int
    matched: int
    warnings: int
    criticals: int
    total_variance_usd: float


class BillingReconciler:
    """Reconciles a set of BillingRow objects against cost_records."""

    async def reconcile(
        self,
        rows: list[BillingRow],
        date_from: date,
        date_to: date,
        session_factory,
        uploaded_by: str,
        team_id: str,
        audit_logger=None,
    ) -> ReconciliationSummary:
        """Run reconciliation and write results to billing_reconciliations.

        Args:
            rows: Parsed billing rows from the CSV upload.
            date_from: Inclusive start of reconciliation window (for logging).
            date_to: Inclusive end of reconciliation window.
            session_factory: Async SQLAlchemy session factory.
            uploaded_by: Actor sub (JWT sub claim) performing the upload.
            team_id: Team performing the upload; used for scoping queries.
            audit_logger: Optional AuditLogger instance; skipped if None.

        Returns:
            ReconciliationSummary with counts and total variance.
        """
        if not rows:
            return ReconciliationSummary(0, 0, 0, 0, 0.0)

        # Fetch Tidus costs for the date range from cost_records
        date_start_dt = datetime(date_from.year, date_from.month, date_from.day, tzinfo=UTC)
        date_end_dt = datetime(date_to.year, date_to.month, date_to.day, tzinfo=UTC) + timedelta(days=1)

        # Fetch individual records and aggregate in Python to avoid
        # CAST(DateTime AS Date) dialect inconsistencies with aiosqlite.
        async with session_factory() as session:
            result = await session.execute(
                select(
                    CostRecordORM.model_id,
                    CostRecordORM.timestamp,
                    CostRecordORM.cost_usd,
                )
                .where(
                    CostRecordORM.team_id == team_id,
                    CostRecordORM.timestamp >= date_start_dt,
                    CostRecordORM.timestamp < date_end_dt,
                )
            )
            tidus_costs: dict[tuple[str, date], float] = {}
            for r in result.all():
                ts = r.timestamp
                d: date = ts.date() if isinstance(ts, datetime) else ts
                key = (r.model_id, d)
                tidus_costs[key] = tidus_costs.get(key, 0.0) + r.cost_usd

        # Write reconciliation rows
        reconciliations: list[BillingReconciliationORM] = []
        for billing_row in rows:
            key = (billing_row.model_id, billing_row.date)
            tidus_cost = tidus_costs.get(key, 0.0)
            provider_cost = billing_row.provider_cost_usd

            variance_usd = provider_cost - tidus_cost
            # Avoid division by zero: if provider charged $0, any Tidus cost is 100% variance
            if provider_cost == 0.0:
                variance_pct = 0.0 if tidus_cost == 0.0 else 1.0
            else:
                variance_pct = variance_usd / provider_cost

            status = _classify_status(variance_pct)
            notes = None
            if key not in tidus_costs:
                notes = "model_id not found in cost_records for this date"

            reconciliations.append(BillingReconciliationORM(
                id=str(uuid.uuid4()),
                reconciliation_date=billing_row.date,
                uploaded_by=uploaded_by,
                team_id=team_id,
                model_id=billing_row.model_id,
                tidus_cost_usd=tidus_cost,
                provider_cost_usd=provider_cost,
                variance_usd=round(variance_usd, 6),
                variance_pct=round(variance_pct, 6),
                status=status,
                notes=notes,
            ))

        async with session_factory() as session:
            session.add_all(reconciliations)
            await session.commit()

        matched = sum(1 for r in reconciliations if r.status == "matched")
        warnings = sum(1 for r in reconciliations if r.status == "warning")
        criticals = sum(1 for r in reconciliations if r.status == "critical")
        total_variance = sum(r.variance_usd for r in reconciliations)

        summary = ReconciliationSummary(
            reconciliation_count=len(reconciliations),
            matched=matched,
            warnings=warnings,
            criticals=criticals,
            total_variance_usd=round(total_variance, 6),
        )

        if audit_logger is not None:
            try:
                from tidus.auth.middleware import TokenPayload
                _actor = TokenPayload(
                    sub=uploaded_by,
                    team_id=team_id,
                    role="team_manager",
                    permissions=[],
                    raw_claims={},
                )
                await audit_logger.record(
                    actor=_actor,
                    action="billing.reconcile",
                    resource_type="billing_reconciliation",
                    resource_id=f"{date_from}/{date_to}",
                    metadata={
                        "team_id": team_id,
                        "rows": len(reconciliations),
                        "criticals": criticals,
                        "warnings": warnings,
                        "total_variance_usd": summary.total_variance_usd,
                    },
                )
            except Exception as exc:
                log.warning("billing_audit_failed", error=str(exc))

        log.info(
            "billing_reconciliation_complete",
            team_id=team_id,
            rows=len(reconciliations),
            matched=matched,
            warnings=warnings,
            criticals=criticals,
        )
        return summary
