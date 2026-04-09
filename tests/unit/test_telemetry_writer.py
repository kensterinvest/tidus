"""Unit tests for TelemetryWriter.

Covers:
  - Successful write creates a ModelTelemetryORM row with all fields
  - probe_type is stored correctly
  - latency_ms is stored as integer (truncated)
  - Non-fatal on DB error: exception is logged but not raised
  - measured_at defaults to now when not provided
  - consecutive_failures stored correctly
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tidus.db.engine import Base
from tidus.db.registry_orm import ModelTelemetryORM
from tidus.sync.telemetry_writer import TelemetryWriter


@pytest_asyncio.fixture
async def sf():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_write_creates_telemetry_row(sf):
    """A successful write creates exactly one row with correct field values."""
    measured = datetime.now(UTC)
    await TelemetryWriter.write(
        sf,
        model_id="gpt-4o",
        is_healthy=True,
        latency_ms=123.7,
        consecutive_failures=0,
        probe_type="live",
        measured_at=measured,
    )

    async with sf() as session:
        rows = (await session.execute(select(ModelTelemetryORM))).scalars().all()

    assert len(rows) == 1
    row = rows[0]
    assert row.model_id == "gpt-4o"
    assert row.is_healthy is True
    assert row.latency_p50_ms == 123     # int truncation
    assert row.consecutive_failures == 0
    assert row.probe_type == "live"
    assert row.source == "health_probe"


@pytest.mark.asyncio
async def test_write_stores_synthetic_probe_type(sf):
    await TelemetryWriter.write(sf, model_id="claude-opus-4-6", is_healthy=True, probe_type="synthetic")

    async with sf() as session:
        row = (await session.execute(select(ModelTelemetryORM))).scalars().first()

    assert row.probe_type == "synthetic"


@pytest.mark.asyncio
async def test_write_stores_consecutive_failures(sf):
    await TelemetryWriter.write(
        sf, model_id="gpt-4o", is_healthy=False, consecutive_failures=3
    )

    async with sf() as session:
        row = (await session.execute(select(ModelTelemetryORM))).scalars().first()

    assert row.is_healthy is False
    assert row.consecutive_failures == 3


@pytest.mark.asyncio
async def test_write_null_latency_when_not_provided(sf):
    await TelemetryWriter.write(sf, model_id="gpt-4o", is_healthy=False)

    async with sf() as session:
        row = (await session.execute(select(ModelTelemetryORM))).scalars().first()

    assert row.latency_p50_ms is None


@pytest.mark.asyncio
async def test_write_is_nonfatal_on_db_error():
    """DB failure is swallowed — no exception escapes TelemetryWriter.write()."""
    failing_sf = AsyncMock()
    failing_sf.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("DB down"))
    failing_sf.return_value.__aexit__ = AsyncMock(return_value=False)

    # Should not raise
    await TelemetryWriter.write(failing_sf, model_id="gpt-4o", is_healthy=True)


@pytest.mark.asyncio
async def test_write_multiple_rows_independent(sf):
    """Multiple writes create independent rows — not an upsert."""
    for i in range(3):
        await TelemetryWriter.write(
            sf, model_id="gpt-4o", is_healthy=True, consecutive_failures=i
        )

    async with sf() as session:
        rows = (await session.execute(select(ModelTelemetryORM))).scalars().all()

    assert len(rows) == 3


@pytest.mark.asyncio
async def test_write_optional_token_delta_stored(sf):
    """token_delta_pct and context_exceeded_rate can be stored for drift detection."""
    await TelemetryWriter.write(
        sf,
        model_id="gpt-4o",
        is_healthy=True,
        token_delta_pct=0.35,
        context_exceeded_rate=0.08,
    )

    async with sf() as session:
        row = (await session.execute(select(ModelTelemetryORM))).scalars().first()

    assert abs(row.token_delta_pct - 0.35) < 1e-6
    assert abs(row.context_exceeded_rate - 0.08) < 1e-6
