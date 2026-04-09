"""Unit tests for BillingReconciler.

Covers:
  - Matched case: |variance_pct| ≤ 5%
  - Warning case: 5% < |variance_pct| ≤ 25%
  - Critical case: |variance_pct| > 25%
  - Zero provider cost edge case
  - Missing model_id in cost_records (tidus_cost = 0, notes set)
  - Date range filtering (only rows within range included)
  - Multiple models with mixed statuses
  - Empty rows list returns zero-count summary
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tidus.billing.csv_parser import BillingRow
from tidus.billing.reconciler import BillingReconciler
from tidus.db.engine import Base, CostRecordORM
from tidus.db.registry_orm import BillingReconciliationORM


@pytest_asyncio.fixture
async def sf():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def _insert_cost(sf, model_id: str, cost_usd: float, team_id: str = "team-a", day: date = date(2026, 4, 1)):
    ts = datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=UTC)
    async with sf() as session:
        session.add(CostRecordORM(
            id=str(uuid.uuid4()),
            task_id="task-1",
            team_id=team_id,
            routing_decision_id="rd-1",
            model_id=model_id,
            vendor="openai",
            input_tokens=100,
            output_tokens=50,
            cost_usd=cost_usd,
            latency_ms=200.0,
            timestamp=ts,
        ))
        await session.commit()


def _billing_row(model_id: str, provider_cost: float, day: date = date(2026, 4, 1)) -> BillingRow:
    return BillingRow(model_id=model_id, date=day, provider_cost_usd=provider_cost)


@pytest.mark.asyncio
async def test_matched_case(sf):
    """When Tidus cost ≈ provider cost (within 5%), status is 'matched'."""
    await _insert_cost(sf, "gpt-4o", cost_usd=100.0)
    rows = [_billing_row("gpt-4o", provider_cost=101.0)]  # variance = 1/101 ≈ 0.99% < 5%

    summary = await BillingReconciler().reconcile(
        rows, date(2026, 4, 1), date(2026, 4, 1), sf, "user-1", "team-a"
    )

    assert summary.reconciliation_count == 1
    assert summary.matched == 1
    assert summary.warnings == 0
    assert summary.criticals == 0

    async with sf() as session:
        row = (await session.execute(select(BillingReconciliationORM))).scalars().first()
    assert row.status == "matched"


@pytest.mark.asyncio
async def test_warning_case(sf):
    """When |variance_pct| is between 5% and 25%, status is 'warning'."""
    await _insert_cost(sf, "gpt-4o", cost_usd=80.0)
    rows = [_billing_row("gpt-4o", provider_cost=100.0)]  # variance = 20/100 = 20% → warning

    summary = await BillingReconciler().reconcile(
        rows, date(2026, 4, 1), date(2026, 4, 1), sf, "user-1", "team-a"
    )

    assert summary.warnings == 1
    assert summary.matched == 0
    async with sf() as session:
        row = (await session.execute(select(BillingReconciliationORM))).scalars().first()
    assert row.status == "warning"


@pytest.mark.asyncio
async def test_critical_case(sf):
    """When |variance_pct| > 25%, status is 'critical'."""
    await _insert_cost(sf, "gpt-4o", cost_usd=50.0)
    rows = [_billing_row("gpt-4o", provider_cost=100.0)]  # variance = 50/100 = 50% → critical

    summary = await BillingReconciler().reconcile(
        rows, date(2026, 4, 1), date(2026, 4, 1), sf, "user-1", "team-a"
    )

    assert summary.criticals == 1
    async with sf() as session:
        row = (await session.execute(select(BillingReconciliationORM))).scalars().first()
    assert row.status == "critical"


@pytest.mark.asyncio
async def test_zero_provider_cost_both_zero(sf):
    """Provider and Tidus both $0 → variance_pct=0 → matched."""
    # No cost_records inserted → tidus_cost=0
    rows = [_billing_row("gpt-4o", provider_cost=0.0)]

    summary = await BillingReconciler().reconcile(
        rows, date(2026, 4, 1), date(2026, 4, 1), sf, "user-1", "team-a"
    )

    assert summary.matched == 1
    async with sf() as session:
        row = (await session.execute(select(BillingReconciliationORM))).scalars().first()
    assert row.status == "matched"
    assert row.variance_pct == 0.0


@pytest.mark.asyncio
async def test_missing_model_in_cost_records(sf):
    """Model not in cost_records → tidus_cost=0, notes populated, status likely critical."""
    rows = [_billing_row("unknown-model", provider_cost=100.0)]

    summary = await BillingReconciler().reconcile(
        rows, date(2026, 4, 1), date(2026, 4, 1), sf, "user-1", "team-a"
    )

    assert summary.criticals == 1
    async with sf() as session:
        row = (await session.execute(select(BillingReconciliationORM))).scalars().first()
    assert row.tidus_cost_usd == 0.0
    assert "not found" in (row.notes or "")


@pytest.mark.asyncio
async def test_date_range_excludes_out_of_range_records(sf):
    """Cost records outside the date range are excluded from Tidus costs."""
    # Insert cost on Apr 3 — but we reconcile for Apr 1 only
    await _insert_cost(sf, "gpt-4o", cost_usd=90.0, day=date(2026, 4, 3))
    rows = [_billing_row("gpt-4o", provider_cost=100.0, day=date(2026, 4, 1))]

    summary = await BillingReconciler().reconcile(
        rows, date(2026, 4, 1), date(2026, 4, 1), sf, "user-1", "team-a"
    )

    # tidus_cost for Apr 1 = 0 (only Apr 3 record exists) → critical
    assert summary.criticals == 1
    async with sf() as session:
        row = (await session.execute(select(BillingReconciliationORM))).scalars().first()
    assert row.tidus_cost_usd == 0.0


@pytest.mark.asyncio
async def test_empty_rows_returns_zero_summary(sf):
    summary = await BillingReconciler().reconcile(
        [], date(2026, 4, 1), date(2026, 4, 1), sf, "user-1", "team-a"
    )
    assert summary.reconciliation_count == 0


@pytest.mark.asyncio
async def test_multiple_models_mixed_statuses(sf):
    """Reconciliation with three models: one matched, one warning, one critical."""
    await _insert_cost(sf, "model-a", cost_usd=100.0)   # provider=101  → matched
    await _insert_cost(sf, "model-b", cost_usd=80.0)    # provider=100  → warning (20%)
    # model-c has no cost record                         # provider=100  → critical (100%)

    rows = [
        _billing_row("model-a", 101.0),
        _billing_row("model-b", 100.0),
        _billing_row("model-c", 100.0),
    ]

    summary = await BillingReconciler().reconcile(
        rows, date(2026, 4, 1), date(2026, 4, 1), sf, "user-1", "team-a"
    )

    assert summary.reconciliation_count == 3
    assert summary.matched == 1
    assert summary.warnings == 1
    assert summary.criticals == 1
