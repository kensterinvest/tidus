"""Unit tests for DriftEngine._auto_resolve_stale_warnings().

Tests that:
  - Warning events older than 72h and NOT in current_detections are auto-resolved.
  - Warning events that ARE in current_detections (still active) are left open.
  - Critical events are never touched by this method.
  - Fresh warning events (< 72h) are not resolved.
  - Batch update writes drift_status='auto_resolved' and resolved_at for eligible rows.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from tidus.db.registry_orm import Base, ModelDriftEventORM
from tidus.sync.drift.engine import DriftEngine

# ── In-memory SQLite fixture ───────────────────────────────────────────────────

@pytest.fixture
async def sf():
    """Async session factory backed by an in-memory SQLite database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def _session_factory():
        async with factory() as session:
            yield session

    yield _session_factory
    await engine.dispose()


async def _insert_drift_event(
    sf,
    model_id: str,
    drift_type: str,
    severity: str,
    drift_status: str = "open",
    age_hours: float = 0,
    resolved_at=None,
) -> str:
    """Insert a ModelDriftEventORM row and return its id."""
    event_id = str(uuid.uuid4())
    detected_at = datetime.now(UTC) - timedelta(hours=age_hours)
    async with sf() as session:
        session.add(ModelDriftEventORM(
            id=event_id,
            model_id=model_id,
            drift_type=drift_type,
            severity=severity,
            metric_value=0.1,
            threshold_value=0.05,
            drift_status=drift_status,
            detected_at=detected_at,
            resolved_at=resolved_at,
        ))
        await session.commit()
    return event_id


# ── _auto_resolve_stale_warnings tests ────────────────────────────────────────

class TestAutoResolveStaleWarnings:

    def _engine(self, sf):
        return DriftEngine(session_factory=sf, registry=None, override_manager=None)

    @pytest.mark.asyncio
    async def test_old_warning_not_in_detections_is_resolved(self, sf):
        """Warning event older than 72h and absent from current_detections → auto_resolved."""
        event_id = await _insert_drift_event(
            sf, "model-a", "latency", "warning", drift_status="open", age_hours=80,
        )
        engine = self._engine(sf)
        await engine._auto_resolve_stale_warnings(set())  # empty: no active detections

        async with sf() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ModelDriftEventORM).where(ModelDriftEventORM.id == event_id)
            )
            row = result.scalars().first()

        assert row.drift_status == "auto_resolved"
        assert row.resolved_at is not None

    @pytest.mark.asyncio
    async def test_warning_still_detected_this_cycle_is_not_resolved(self, sf):
        """Warning event still active in current detections is left open."""
        event_id = await _insert_drift_event(
            sf, "model-b", "latency", "warning", drift_status="open", age_hours=100,
        )
        engine = self._engine(sf)
        # Pass the key as currently detected → should NOT be resolved
        await engine._auto_resolve_stale_warnings({("model-b", "latency")})

        async with sf() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ModelDriftEventORM).where(ModelDriftEventORM.id == event_id)
            )
            row = result.scalars().first()

        assert row.drift_status == "open", "Active detection should remain open"

    @pytest.mark.asyncio
    async def test_fresh_warning_not_resolved(self, sf):
        """Warning event younger than 72h is not resolved even if absent from detections."""
        event_id = await _insert_drift_event(
            sf, "model-c", "context", "warning", drift_status="open", age_hours=10,
        )
        engine = self._engine(sf)
        await engine._auto_resolve_stale_warnings(set())

        async with sf() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ModelDriftEventORM).where(ModelDriftEventORM.id == event_id)
            )
            row = result.scalars().first()

        assert row.drift_status == "open", "Fresh warning should not be auto-resolved"

    @pytest.mark.asyncio
    async def test_critical_events_never_auto_resolved_by_this_method(self, sf):
        """_auto_resolve_stale_warnings must never touch critical events."""
        event_id = await _insert_drift_event(
            sf, "model-d", "latency", "critical", drift_status="open", age_hours=200,
        )
        engine = self._engine(sf)
        await engine._auto_resolve_stale_warnings(set())

        async with sf() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ModelDriftEventORM).where(ModelDriftEventORM.id == event_id)
            )
            row = result.scalars().first()

        assert row.drift_status == "open", "Critical events must not be touched by this method"

    @pytest.mark.asyncio
    async def test_batch_resolves_multiple_stale_warnings(self, sf):
        """Multiple stale warning events are all resolved in a single call."""
        ids = []
        for model_id in ("batch-a", "batch-b", "batch-c"):
            eid = await _insert_drift_event(
                sf, model_id, "price", "warning", drift_status="open", age_hours=100,
            )
            ids.append(eid)

        engine = self._engine(sf)
        await engine._auto_resolve_stale_warnings(set())

        async with sf() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ModelDriftEventORM).where(ModelDriftEventORM.id.in_(ids))
            )
            rows = result.scalars().all()

        assert all(r.drift_status == "auto_resolved" for r in rows)
        assert all(r.resolved_at is not None for r in rows)

    @pytest.mark.asyncio
    async def test_already_resolved_events_not_reprocessed(self, sf):
        """Events already in auto_resolved or manually_resolved state are ignored."""
        resolved_id = await _insert_drift_event(
            sf, "model-e", "latency", "warning", drift_status="auto_resolved", age_hours=100,
        )
        engine = self._engine(sf)
        # Should run without error and not crash on already-resolved rows
        await engine._auto_resolve_stale_warnings(set())

        async with sf() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ModelDriftEventORM).where(ModelDriftEventORM.id == resolved_id)
            )
            row = result.scalars().first()

        # Status unchanged — the method filters on drift_status='open'
        assert row.drift_status == "auto_resolved"
