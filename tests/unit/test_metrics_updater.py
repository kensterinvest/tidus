"""Tests for MetricsUpdater — verifies all 6 Gauges are updated from DB and registry state."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_factory(revision=None, completed_at=None):
    """Build a minimal async session_factory mock for MetricsUpdater tests."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    revision_result = MagicMock()
    revision_result.scalars.return_value.first.return_value = revision

    sync_ts_result = MagicMock()
    sync_ts_result.scalar_one_or_none.return_value = completed_at

    # execute() returns different mocks depending on call order
    session.execute = AsyncMock(side_effect=[revision_result, sync_ts_result])

    factory = MagicMock(return_value=session)
    return factory


def _make_spec(model_id: str, last_price_check=None):
    spec = MagicMock()
    spec.model_id = model_id
    spec.last_price_check = last_price_check
    return spec


def _make_registry(*specs):
    registry = MagicMock()
    registry.list_all.return_value = list(specs)
    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_revision_metrics_sets_gauges():
    """Active revision sets both REGISTRY_ACTIVE_REVISION_ID and REGISTRY_ACTIVE_REVISION_TS."""
    from tidus.observability.registry_metrics import (
        REGISTRY_ACTIVE_REVISION_ID,
        REGISTRY_ACTIVE_REVISION_TS,
    )

    revision_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    activated_at = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)

    revision = MagicMock()
    revision.revision_id = revision_id
    revision.activated_at = activated_at
    revision.status = "active"

    factory = _make_session_factory(revision=revision, completed_at=None)
    _make_registry()

    with patch.object(REGISTRY_ACTIVE_REVISION_ID, "set") as mock_rev_id, \
         patch.object(REGISTRY_ACTIVE_REVISION_TS, "set") as mock_rev_ts:

        from tidus.observability.metrics_updater import MetricsUpdater
        updater = MetricsUpdater()
        await updater._update_revision_metrics(factory)

    mock_rev_id.assert_called_once()
    mock_rev_ts.assert_called_once_with(activated_at.timestamp())


@pytest.mark.asyncio
async def test_update_revision_metrics_skips_when_no_revision():
    """No active revision → Gauge is not updated."""
    from tidus.observability.registry_metrics import REGISTRY_ACTIVE_REVISION_ID

    factory = _make_session_factory(revision=None, completed_at=None)

    with patch.object(REGISTRY_ACTIVE_REVISION_ID, "set") as mock_set:
        from tidus.observability.metrics_updater import MetricsUpdater
        await MetricsUpdater()._update_revision_metrics(factory)

    mock_set.assert_not_called()


@pytest.mark.asyncio
async def test_update_sync_timestamp_sets_gauge():
    """Successful ingestion run sets REGISTRY_LAST_SYNC_TS."""
    from tidus.observability.registry_metrics import REGISTRY_LAST_SYNC_TS

    completed_at = datetime(2026, 4, 1, 3, 0, 0, tzinfo=UTC)

    # Build a session whose execute() returns a result with scalar_one_or_none → completed_at
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    result = MagicMock()
    result.scalar_one_or_none.return_value = completed_at
    session.execute = AsyncMock(return_value=result)
    factory = MagicMock(return_value=session)

    with patch.object(REGISTRY_LAST_SYNC_TS, "set") as mock_set:
        from tidus.observability.metrics_updater import MetricsUpdater
        await MetricsUpdater()._update_sync_timestamp(factory)

    mock_set.assert_called_once_with(completed_at.timestamp())


@pytest.mark.asyncio
async def test_update_sync_timestamp_skips_when_none():
    """No successful ingestion run → REGISTRY_LAST_SYNC_TS not updated."""
    from tidus.observability.registry_metrics import REGISTRY_LAST_SYNC_TS

    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)
    factory = MagicMock(return_value=session)

    with patch.object(REGISTRY_LAST_SYNC_TS, "set") as mock_set:
        from tidus.observability.metrics_updater import MetricsUpdater
        await MetricsUpdater()._update_sync_timestamp(factory)

    mock_set.assert_not_called()


def test_update_model_metrics_fresh_price():
    """Model with recent price check → confidence=1.0, not counted as stale."""
    from tidus.observability.registry_metrics import (
        REGISTRY_MODEL_CONFIDENCE,
        REGISTRY_STALE_MODEL_COUNT,
    )

    fresh_date = date.today()
    spec = _make_spec("gpt-4o", last_price_check=fresh_date)
    registry = _make_registry(spec)

    with patch.object(REGISTRY_MODEL_CONFIDENCE, "labels") as mock_conf_labels, \
         patch.object(REGISTRY_STALE_MODEL_COUNT, "set") as mock_stale:
        mock_conf_child = MagicMock()
        mock_conf_labels.return_value = mock_conf_child

        from tidus.observability.metrics_updater import MetricsUpdater
        MetricsUpdater()._update_model_metrics(registry)

    mock_conf_labels.assert_called_once_with(model_id="gpt-4o")
    mock_conf_child.set.assert_called_once_with(1.0)
    mock_stale.assert_called_once_with(0)


def test_update_model_metrics_stale_price():
    """Model with old price check → confidence=0.5, counted as stale."""
    from tidus.observability.registry_metrics import (
        REGISTRY_MODEL_CONFIDENCE,
        REGISTRY_STALE_MODEL_COUNT,
    )

    stale_date = date.today() - timedelta(days=10)
    spec = _make_spec("old-model", last_price_check=stale_date)
    registry = _make_registry(spec)

    with patch.object(REGISTRY_MODEL_CONFIDENCE, "labels") as mock_conf_labels, \
         patch.object(REGISTRY_STALE_MODEL_COUNT, "set") as mock_stale:
        mock_conf_child = MagicMock()
        mock_conf_labels.return_value = mock_conf_child

        from tidus.observability.metrics_updater import MetricsUpdater
        MetricsUpdater()._update_model_metrics(registry)

    mock_conf_child.set.assert_called_once_with(0.5)
    mock_stale.assert_called_once_with(1)


def test_update_model_metrics_no_price():
    """Model with no last_price_check → confidence=0.5, counted as stale."""
    from tidus.observability.registry_metrics import (
        REGISTRY_MODEL_CONFIDENCE,
        REGISTRY_STALE_MODEL_COUNT,
    )

    spec = _make_spec("no-price-model", last_price_check=None)
    registry = _make_registry(spec)

    with patch.object(REGISTRY_MODEL_CONFIDENCE, "labels") as mock_conf_labels, \
         patch.object(REGISTRY_STALE_MODEL_COUNT, "set") as mock_stale:
        mock_conf_child = MagicMock()
        mock_conf_labels.return_value = mock_conf_child

        from tidus.observability.metrics_updater import MetricsUpdater
        MetricsUpdater()._update_model_metrics(registry)

    mock_conf_child.set.assert_called_once_with(0.5)
    mock_stale.assert_called_once_with(1)


def test_update_model_metrics_stale_count_multiple():
    """Mixed fresh and stale models → correct stale count."""
    from tidus.observability.registry_metrics import REGISTRY_STALE_MODEL_COUNT

    fresh = _make_spec("fresh-model", last_price_check=date.today())
    stale1 = _make_spec("stale-1", last_price_check=date.today() - timedelta(days=15))
    stale2 = _make_spec("stale-2", last_price_check=None)
    registry = _make_registry(fresh, stale1, stale2)

    with patch.object(REGISTRY_STALE_MODEL_COUNT, "set") as mock_stale, \
         patch("tidus.observability.metrics_updater.REGISTRY_MODEL_CONFIDENCE"):
        from tidus.observability.metrics_updater import MetricsUpdater
        MetricsUpdater()._update_model_metrics(registry)

    mock_stale.assert_called_once_with(2)


def test_update_model_metrics_none_registry():
    """None registry → no-op, no error."""
    from tidus.observability.metrics_updater import MetricsUpdater

    # Should not raise
    MetricsUpdater()._update_model_metrics(None)


@pytest.mark.asyncio
async def test_full_update_is_non_fatal():
    """update() catches exceptions from sub-methods and logs, never raises."""
    from tidus.observability.metrics_updater import MetricsUpdater

    # session_factory raises immediately
    async def bad_factory():
        raise RuntimeError("DB down")

    factory = MagicMock(side_effect=RuntimeError("DB down"))
    registry = MagicMock()
    registry.list_all.return_value = []

    # Should not raise
    await MetricsUpdater().update(registry, factory)
