"""Integration tests for HealthProbe telemetry persistence.

Covers:
  - probe.run_once() with a session_factory writes telemetry rows
  - TelemetryReader reads the persisted rows after the probe completes
  - HealthProbe re-reads persisted data on a second run (simulated restart)
  - Synthetic probe writes probe_type='synthetic', no live call charged
  - Tier A model (consecutive_failures > 0) writes probe_type='live'
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tidus.db.engine import Base
from tidus.db.registry_orm import ModelTelemetryORM
from tidus.models.model_registry import ModelSpec
from tidus.registry.telemetry_reader import TelemetryReader
from tidus.sync.health_probe import HealthProbe


@pytest_asyncio.fixture
async def sf():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _make_spec(model_id: str = "gpt-4o") -> ModelSpec:
    return ModelSpec.model_validate({
        "model_id": model_id, "vendor": "openai", "tier": 2,
        "max_context": 128_000, "input_price": 5.0, "output_price": 10.0,
        "tokenizer": "tiktoken_o200k",
    })


def _mock_registry(*specs: ModelSpec):
    registry = MagicMock()
    registry.list_enabled.return_value = list(specs)
    registry.update_latency = MagicMock()
    registry.set_enabled = MagicMock()
    return registry


def _mock_adapter(healthy: bool = True, count_tokens_ok: bool = True):
    adapter = MagicMock()
    adapter.health_check = AsyncMock(return_value=healthy)
    if count_tokens_ok:
        adapter.count_tokens = AsyncMock(return_value=10)
    else:
        adapter.count_tokens = AsyncMock(side_effect=Exception("tokenize failed"))
    return adapter


async def _insert_telemetry(sf, model_id: str, consecutive_failures: int, age_minutes: int = 5):
    measured = datetime.now(UTC) - timedelta(minutes=age_minutes)
    async with sf() as session:
        session.add(ModelTelemetryORM(
            id=str(uuid.uuid4()),
            model_id=model_id,
            measured_at=measured,
            is_healthy=consecutive_failures == 0,
            consecutive_failures=consecutive_failures,
            source="health_probe",
        ))
        await session.commit()


# ── Probe writes telemetry to DB ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_probe_writes_telemetry_row(sf):
    """run_once() with a session_factory creates a ModelTelemetryORM row."""
    spec = _make_spec("gpt-4o")
    adapter = _mock_adapter(healthy=True)
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        probe = HealthProbe(registry, session_factory=sf)
        await probe.run_once()

    async with sf() as session:
        rows = (await session.execute(select(ModelTelemetryORM))).scalars().all()

    assert len(rows) == 1
    assert rows[0].model_id == "gpt-4o"
    assert rows[0].is_healthy is True


# ── TelemetryReader reads probe output ───────────────────────────────────────

@pytest.mark.asyncio
async def test_telemetry_reader_reads_probe_output(sf):
    """After a probe run, TelemetryReader.get_all_snapshots() returns the written row."""
    spec = _make_spec("gpt-4o")
    adapter = _mock_adapter(healthy=True)
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        probe = HealthProbe(registry, session_factory=sf)
        await probe.run_once()

    snapshots = await TelemetryReader().get_all_snapshots(sf)
    assert "gpt-4o" in snapshots
    snap = snapshots["gpt-4o"]
    assert snap.is_healthy is True
    assert snap.consecutive_failures == 0
    assert snap.staleness == "fresh"


# ── Simulated restart re-reads persisted data ────────────────────────────────

@pytest.mark.asyncio
async def test_second_probe_instance_reads_persisted_telemetry(sf):
    """A fresh HealthProbe instance reads existing DB telemetry for tier classification."""
    # Pre-seed telemetry for a failing model (consecutive_failures=2 → Tier A)
    await _insert_telemetry(sf, "failing-model", consecutive_failures=2, age_minutes=5)

    spec = _make_spec("failing-model")
    adapter = _mock_adapter(healthy=False)
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        # Brand-new probe instance — no in-memory state, reads from DB
        probe = HealthProbe(registry, session_factory=sf)
        await probe.run_once()

    # Tier A means live probe was used
    adapter.health_check.assert_called_once()
    adapter.count_tokens.assert_not_called()


# ── Synthetic probe logged without live call ──────────────────────────────────

@pytest.mark.asyncio
async def test_synthetic_probe_written_without_live_call(sf):
    """Tier B model uses count_tokens (synthetic) and writes probe_type='synthetic'."""
    spec = _make_spec("new-model")
    # No telemetry → Tier B
    adapter = _mock_adapter(healthy=True, count_tokens_ok=True)
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        probe = HealthProbe(registry, session_factory=sf)
        await probe.run_once()

    # count_tokens called; health_check never charged
    adapter.count_tokens.assert_called_once()
    adapter.health_check.assert_not_called()

    async with sf() as session:
        row = (await session.execute(select(ModelTelemetryORM))).scalars().first()

    assert row is not None
    assert row.probe_type == "synthetic"
    assert row.model_id == "new-model"


# ── Tier A writes probe_type='live' ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_tier_a_probe_writes_live_type(sf):
    """Models with consecutive_failures > 0 use live probe → probe_type='live' in DB."""
    await _insert_telemetry(sf, "struggling-model", consecutive_failures=1, age_minutes=3)

    spec = _make_spec("struggling-model")
    adapter = _mock_adapter(healthy=True)
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        probe = HealthProbe(registry, session_factory=sf)
        await probe.run_once()

    async with sf() as session:
        rows = (await session.execute(
            select(ModelTelemetryORM)
            .where(ModelTelemetryORM.model_id == "struggling-model")
            .order_by(ModelTelemetryORM.measured_at.desc())
        )).scalars().all()

    # The newest row should be from our probe run — probe_type='live'
    assert rows[0].probe_type == "live"
    adapter.health_check.assert_called_once()


# ── No session_factory → no telemetry rows ───────────────────────────────────

@pytest.mark.asyncio
async def test_no_session_factory_no_rows_written(sf):
    """Without session_factory, probe works but writes nothing to DB."""
    spec = _make_spec("gpt-4o")
    adapter = _mock_adapter(healthy=True)
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        probe = HealthProbe(registry, session_factory=None)
        results = await probe.run_once()

    # Probe still returns results
    assert "gpt-4o" in results

    # But DB is empty
    async with sf() as session:
        rows = (await session.execute(select(ModelTelemetryORM))).scalars().all()
    assert rows == []
