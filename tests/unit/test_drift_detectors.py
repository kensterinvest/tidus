"""Unit tests for the four drift detectors.

Covers:
  - Each detector returns no detections when metrics are below threshold
  - Each detector returns 'warning' when above warning threshold
  - Each detector returns 'critical' when above critical threshold
  - Drift detections carry the active_revision_id passed by the caller
  - Empty DB returns no detections (graceful no-data handling)
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tidus.db.engine import Base, CostRecordORM, PriceChangeLogORM
from tidus.db.registry_orm import ModelTelemetryORM
from tidus.models.model_registry import ModelSpec, ModelTier, TokenizerType
from tidus.sync.drift.detectors import (
    ContextDriftDetector,
    LatencyDriftDetector,
    PriceDriftDetector,
    TokenizationDriftDetector,
)


@pytest_asyncio.fixture
async def sf():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _make_spec(
    model_id: str = "gpt-4o",
    latency_p50_ms: int = 1000,
    max_context: int = 128_000,
) -> ModelSpec:
    return ModelSpec.model_validate({
        "model_id": model_id,
        "vendor": "openai",
        "tier": 2,
        "max_context": max_context,
        "input_price": 5.0,
        "output_price": 10.0,
        "tokenizer": "tiktoken_o200k",
        "latency_p50_ms": latency_p50_ms,
    })


def _now() -> datetime:
    return datetime.now(UTC)


# ── LatencyDriftDetector ──────────────────────────────────────────────────────

class TestLatencyDriftDetector:
    async def _insert_telemetry(self, sf, model_id: str, latency_ms: int):
        async with sf() as session:
            session.add(ModelTelemetryORM(
                id=str(uuid.uuid4()),
                model_id=model_id,
                measured_at=_now(),
                latency_p50_ms=latency_ms,
                is_healthy=True,
                consecutive_failures=0,
                source="health_probe",
            ))
            await session.commit()

    @pytest.mark.asyncio
    async def test_no_detection_below_warning(self, sf):
        await self._insert_telemetry(sf, "gpt-4o", 1200)  # ratio=1.2 < 1.5
        spec = _make_spec(latency_p50_ms=1000)
        detections = await LatencyDriftDetector().detect(sf, [spec])
        assert detections == []

    @pytest.mark.asyncio
    async def test_warning_at_1_5x(self, sf):
        await self._insert_telemetry(sf, "gpt-4o", 1600)  # ratio=1.6 >= 1.5
        spec = _make_spec(latency_p50_ms=1000)
        detections = await LatencyDriftDetector().detect(sf, [spec])
        assert len(detections) == 1
        assert detections[0].severity == "warning"
        assert detections[0].drift_type == "latency"

    @pytest.mark.asyncio
    async def test_critical_at_2_5x(self, sf):
        await self._insert_telemetry(sf, "gpt-4o", 2600)  # ratio=2.6 >= 2.5
        spec = _make_spec(latency_p50_ms=1000)
        detections = await LatencyDriftDetector().detect(sf, [spec])
        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_active_revision_id_carried(self, sf):
        await self._insert_telemetry(sf, "gpt-4o", 2600)
        spec = _make_spec(latency_p50_ms=1000)
        detections = await LatencyDriftDetector().detect(sf, [spec], active_revision_id="rev-abc")
        assert detections[0].active_revision_id == "rev-abc"

    @pytest.mark.asyncio
    async def test_empty_db_returns_no_detections(self, sf):
        spec = _make_spec(latency_p50_ms=1000)
        detections = await LatencyDriftDetector().detect(sf, [spec])
        assert detections == []


# ── TokenizationDriftDetector ─────────────────────────────────────────────────

class TestTokenizationDriftDetector:
    async def _insert_telemetry(self, sf, model_id: str, token_delta_pct: float):
        async with sf() as session:
            session.add(ModelTelemetryORM(
                id=str(uuid.uuid4()),
                model_id=model_id,
                measured_at=_now(),
                is_healthy=True,
                consecutive_failures=0,
                token_delta_pct=token_delta_pct,
                source="health_probe",
            ))
            await session.commit()

    @pytest.mark.asyncio
    async def test_no_detection_below_warning(self, sf):
        await self._insert_telemetry(sf, "gpt-4o", 0.10)  # 10% < 25%
        spec = _make_spec()
        detections = await TokenizationDriftDetector().detect(sf, [spec])
        assert detections == []

    @pytest.mark.asyncio
    async def test_warning_at_threshold(self, sf):
        await self._insert_telemetry(sf, "gpt-4o", 0.30)  # 30% >= 25%
        spec = _make_spec()
        detections = await TokenizationDriftDetector().detect(sf, [spec])
        assert len(detections) == 1
        assert detections[0].severity == "warning"
        assert detections[0].drift_type == "tokenization"

    @pytest.mark.asyncio
    async def test_critical_at_threshold(self, sf):
        await self._insert_telemetry(sf, "gpt-4o", 0.60)  # 60% >= 50%
        spec = _make_spec()
        detections = await TokenizationDriftDetector().detect(sf, [spec])
        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_negative_delta_uses_absolute_value(self, sf):
        await self._insert_telemetry(sf, "gpt-4o", -0.30)  # abs = 30% >= 25%
        spec = _make_spec()
        detections = await TokenizationDriftDetector().detect(sf, [spec])
        assert len(detections) == 1
        assert detections[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_no_telemetry_returns_no_detections(self, sf):
        spec = _make_spec()
        detections = await TokenizationDriftDetector().detect(sf, [spec])
        assert detections == []


# ── PriceDriftDetector ────────────────────────────────────────────────────────

class TestPriceDriftDetector:
    async def _insert_price_change(self, sf, model_id: str, delta_pct: float):
        async with sf() as session:
            session.add(PriceChangeLogORM(
                id=str(uuid.uuid4()),
                model_id=model_id,
                vendor="openai",
                field_changed="input_price",
                old_value=5.0,
                new_value=5.0 * (1 + delta_pct),
                delta_pct=delta_pct,
                detected_at=_now(),
                source="price_sync",
            ))
            await session.commit()

    @pytest.mark.asyncio
    async def test_no_detection_below_thresholds(self, sf):
        await self._insert_price_change(sf, "gpt-4o", 0.05)  # 5% < 15%
        spec = _make_spec()
        detections = await PriceDriftDetector().detect(sf, [spec])
        assert detections == []

    @pytest.mark.asyncio
    async def test_warning_on_large_single_change(self, sf):
        await self._insert_price_change(sf, "gpt-4o", 0.20)  # 20% >= 15%
        spec = _make_spec()
        detections = await PriceDriftDetector().detect(sf, [spec])
        assert len(detections) == 1
        assert detections[0].severity == "warning"
        assert detections[0].drift_type == "price"

    @pytest.mark.asyncio
    async def test_warning_on_too_many_changes(self, sf):
        for _ in range(4):  # 4 > max_changes_30d=3
            await self._insert_price_change(sf, "gpt-4o", 0.05)
        spec = _make_spec()
        detections = await PriceDriftDetector().detect(sf, [spec])
        assert len(detections) == 1
        assert detections[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_empty_db_returns_no_detections(self, sf):
        spec = _make_spec()
        detections = await PriceDriftDetector().detect(sf, [spec])
        assert detections == []


# ── ContextDriftDetector ──────────────────────────────────────────────────────

class TestContextDriftDetector:
    """Tests for overflow-rate detection (fraction of requests near the context limit).

    Uses max_context=1000 so overflow_threshold=900 — easy to construct test data.
    """

    async def _insert_cost_record(self, sf, model_id: str, input_tokens: int):
        async with sf() as session:
            session.add(CostRecordORM(
                id=str(uuid.uuid4()),
                task_id="task-1",
                team_id="team-a",
                routing_decision_id="rd-1",
                model_id=model_id,
                vendor="openai",
                input_tokens=input_tokens,
                output_tokens=100,
                cost_usd=0.01,
                latency_ms=200.0,
            ))
            await session.commit()

    @pytest.mark.asyncio
    async def test_no_detection_below_warning(self, sf):
        # 20 requests, none near limit (all < 900) → overflow_rate = 0% < 5%
        for _ in range(20):
            await self._insert_cost_record(sf, "gpt-4o", input_tokens=100)
        spec = _make_spec(max_context=1000)
        detections = await ContextDriftDetector().detect(sf, [spec])
        assert detections == []

    @pytest.mark.asyncio
    async def test_warning_at_threshold(self, sf):
        # 1 overflow (input=950 >= 900) + 19 normal = overflow_rate = 5% >= warning(5%)
        await self._insert_cost_record(sf, "gpt-4o", input_tokens=950)
        for _ in range(19):
            await self._insert_cost_record(sf, "gpt-4o", input_tokens=100)
        spec = _make_spec(max_context=1000)
        detections = await ContextDriftDetector().detect(sf, [spec])
        assert len(detections) == 1
        assert detections[0].severity == "warning"
        assert detections[0].drift_type == "context"

    @pytest.mark.asyncio
    async def test_critical_at_threshold(self, sf):
        # 3 overflows + 17 normal = overflow_rate = 15% >= critical(15%)
        for _ in range(3):
            await self._insert_cost_record(sf, "gpt-4o", input_tokens=950)
        for _ in range(17):
            await self._insert_cost_record(sf, "gpt-4o", input_tokens=100)
        spec = _make_spec(max_context=1000)
        detections = await ContextDriftDetector().detect(sf, [spec])
        assert len(detections) == 1
        assert detections[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_empty_db_returns_no_detections(self, sf):
        spec = _make_spec(max_context=1000)
        detections = await ContextDriftDetector().detect(sf, [spec])
        assert detections == []
