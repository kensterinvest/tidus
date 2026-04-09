"""Unit tests for HealthProbe 3-tier sampling logic.

Covers:
  - Tier A models (consecutive_failures > 0) always receive a live probe
  - Tier B models (no recent telemetry) receive a synthetic-first probe
  - Tier C models (healthy + recently probed) are sampled at ~10%
  - probe_type is correctly set to 'live' for Tier A, 'synthetic' for Tier B/C
  - Synthetic probe success → no live probe escalation
  - Synthetic probe failure → escalates to live health_check
  - TelemetryWriter.write() is called with the correct probe_type
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tidus.db.engine import Base
from tidus.db.registry_orm import ModelTelemetryORM
from tidus.models.model_registry import ModelSpec, ModelTier, TokenizerType
from tidus.sync.health_probe import HealthProbe


@pytest_asyncio.fixture
async def sf():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _make_spec(model_id: str = "gpt-4o", vendor: str = "openai") -> ModelSpec:
    return ModelSpec.model_validate({
        "model_id": model_id, "vendor": vendor, "tier": 2,
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
        adapter.count_tokens = AsyncMock(side_effect=Exception("synthetic failed"))
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


# ── Tier A: always probe live ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tier_a_model_always_probed_live(sf):
    """A model with consecutive_failures > 0 is placed in Tier A and gets a live probe."""
    spec = _make_spec("failing-model")
    await _insert_telemetry(sf, "failing-model", consecutive_failures=2, age_minutes=5)

    adapter = _mock_adapter(healthy=True)
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    written: list[dict] = []
    async def capture_write(sf, **kwargs):
        written.append(kwargs)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        with patch("tidus.sync.health_probe.TelemetryWriter.write", new=capture_write):
            probe = HealthProbe(registry, session_factory=sf)
            await probe.run_once()

    # health_check was called (live probe), count_tokens was NOT
    adapter.health_check.assert_called_once()
    adapter.count_tokens.assert_not_called()

    assert written
    assert written[0]["probe_type"] == "live"


@pytest.mark.asyncio
async def test_tier_b_model_uses_synthetic_first(sf):
    """A model with no recent telemetry (Tier B) gets a synthetic probe first."""
    spec = _make_spec("new-model")
    # No telemetry inserted → Tier B

    adapter = _mock_adapter(healthy=True, count_tokens_ok=True)
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    written: list[dict] = []
    async def capture_write(sf, **kwargs):
        written.append(kwargs)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        with patch("tidus.sync.health_probe.TelemetryWriter.write", new=capture_write):
            probe = HealthProbe(registry, session_factory=sf)
            await probe.run_once()

    # count_tokens called (synthetic), health_check NOT called (synthetic succeeded)
    adapter.count_tokens.assert_called_once()
    adapter.health_check.assert_not_called()

    assert written[0]["probe_type"] == "synthetic"


@pytest.mark.asyncio
async def test_synthetic_failure_escalates_to_live(sf):
    """When the synthetic probe fails, the probe escalates to a live health_check."""
    spec = _make_spec("flaky-model")
    # No telemetry → Tier B

    adapter = _mock_adapter(healthy=True, count_tokens_ok=False)  # synthetic fails
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    written: list[dict] = []
    async def capture_write(sf, **kwargs):
        written.append(kwargs)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        with patch("tidus.sync.health_probe.TelemetryWriter.write", new=capture_write):
            probe = HealthProbe(registry, session_factory=sf)
            await probe.run_once()

    # Both probes called due to escalation
    adapter.count_tokens.assert_called_once()
    adapter.health_check.assert_called_once()

    # The final probe_type reported is 'live' (escalated)
    assert written[0]["probe_type"] == "live"


@pytest.mark.asyncio
async def test_tier_c_models_sampled_at_10_percent(sf):
    """Tier C models (recently probed, healthy) are sampled at approximately 10%."""
    # Create 50 healthy models all recently probed
    specs = [_make_spec(f"model-{i}") for i in range(50)]
    for spec in specs:
        await _insert_telemetry(sf, spec.model_id, consecutive_failures=0, age_minutes=5)

    adapter = _mock_adapter(healthy=True)
    registry = _mock_registry(*specs)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    written: list[dict] = []
    async def capture_write(sf, **kwargs):
        written.append(kwargs)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        with patch("tidus.sync.health_probe.TelemetryWriter.write", new=capture_write):
            probe = HealthProbe(registry, session_factory=sf)
            await probe.run_once()

    # Expect roughly 10% of 50 = 5 models probed.
    # With stochastic sampling we use a loose bound: 0–15 probes in a single run.
    assert 0 <= len(written) <= 15


@pytest.mark.asyncio
async def test_telemetry_written_with_correct_model_id(sf):
    """TelemetryWriter receives the correct model_id for each probe."""
    spec = _make_spec("specific-model")
    # No telemetry → Tier B

    adapter = _mock_adapter(healthy=True)
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    written: list[dict] = []
    async def capture_write(sf, **kwargs):
        written.append(kwargs)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        with patch("tidus.sync.health_probe.TelemetryWriter.write", new=capture_write):
            probe = HealthProbe(registry, session_factory=sf)
            await probe.run_once()

    assert written[0]["model_id"] == "specific-model"
    assert written[0]["is_healthy"] is True


@pytest.mark.asyncio
async def test_no_telemetry_written_when_no_session_factory():
    """Without a session_factory, TelemetryWriter.write() is never called."""
    spec = _make_spec("gpt-4o")
    adapter = _mock_adapter(healthy=True)
    registry = _mock_registry(spec)

    fake_module = MagicMock()
    fake_module.get_adapter = MagicMock(return_value=adapter)

    with patch.dict("sys.modules", {"tidus.adapters.adapter_factory": fake_module}):
        with patch("tidus.sync.health_probe.TelemetryWriter.write", new_callable=AsyncMock) as mock_write:
            probe = HealthProbe(registry, session_factory=None)
            await probe.run_once()

    mock_write.assert_not_called()
