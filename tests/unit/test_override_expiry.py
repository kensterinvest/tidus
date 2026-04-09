"""Unit tests for OverrideExpiryJob.

Covers:
  - Expired overrides are deactivated
  - Non-expired overrides are untouched
  - Audit entries are written for each deactivated override
  - Registry refresh is triggered after deactivation
  - Job handles DB errors gracefully (non-fatal)
  - Returns 0 when nothing expired
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tidus.sync.override_expiry import OverrideExpiryJob


def make_expired_override(override_id="ov-expired", override_type="price_multiplier"):
    o = MagicMock()
    o.override_id = override_id
    o.override_type = override_type
    o.model_id = "gpt-4o"
    o.owner_team_id = "team-a"
    return o


@pytest.mark.asyncio
async def test_expired_overrides_deactivated_and_audit_written():
    expired = [make_expired_override("ov-1"), make_expired_override("ov-2")]
    sf = AsyncMock()

    mock_registry = AsyncMock()
    mock_registry.refresh = AsyncMock(return_value=True)

    with (
        patch(
            "tidus.sync.override_expiry.deactivate_expired_overrides",
            return_value=expired,
        ) as mock_deactivate,
        patch("tidus.sync.override_expiry.AuditLogger") as MockAuditLogger,
    ):
        mock_audit_instance = AsyncMock()
        MockAuditLogger.return_value = mock_audit_instance

        job = OverrideExpiryJob()
        count = await job.run(sf, registry=mock_registry)

    assert count == 2
    mock_deactivate.assert_called_once_with(sf)
    assert mock_audit_instance.record.call_count == 2
    mock_registry.refresh.assert_called_once_with(sf)


@pytest.mark.asyncio
async def test_no_expired_overrides_returns_zero():
    sf = AsyncMock()
    mock_registry = AsyncMock()

    with patch(
        "tidus.sync.override_expiry.deactivate_expired_overrides",
        return_value=[],
    ):
        count = await OverrideExpiryJob().run(sf, registry=mock_registry)

    assert count == 0
    mock_registry.refresh.assert_not_called()


@pytest.mark.asyncio
async def test_registry_refresh_triggered_after_deactivation():
    expired = [make_expired_override()]
    sf = AsyncMock()
    mock_registry = AsyncMock()
    mock_registry.refresh = AsyncMock(return_value=True)

    with (
        patch("tidus.sync.override_expiry.deactivate_expired_overrides", return_value=expired),
        patch("tidus.sync.override_expiry.AuditLogger") as MockAuditLogger,
    ):
        MockAuditLogger.return_value = AsyncMock()
        await OverrideExpiryJob().run(sf, registry=mock_registry)

    mock_registry.refresh.assert_called_once()


@pytest.mark.asyncio
async def test_registry_refresh_not_called_without_registry():
    expired = [make_expired_override()]
    sf = AsyncMock()

    with (
        patch("tidus.sync.override_expiry.deactivate_expired_overrides", return_value=expired),
        patch("tidus.sync.override_expiry.AuditLogger") as MockAuditLogger,
    ):
        MockAuditLogger.return_value = AsyncMock()
        # Pass registry=None explicitly
        count = await OverrideExpiryJob().run(sf, registry=None)

    assert count == 1


@pytest.mark.asyncio
async def test_db_error_is_non_fatal():
    """A DB error during deactivation returns 0 — never raises."""
    sf = AsyncMock()

    with patch(
        "tidus.sync.override_expiry.deactivate_expired_overrides",
        side_effect=Exception("DB connection lost"),
    ):
        count = await OverrideExpiryJob().run(sf)

    assert count == 0


@pytest.mark.asyncio
async def test_audit_entries_use_system_actor():
    """Audit entries written by expiry job must use system_expiry actor, not a user."""
    expired = [make_expired_override("ov-sys")]
    sf = AsyncMock()

    recorded_actors = []

    async def capture_actor(actor, **kwargs):
        recorded_actors.append(actor)

    with (
        patch("tidus.sync.override_expiry.deactivate_expired_overrides", return_value=expired),
        patch("tidus.sync.override_expiry.AuditLogger") as MockAuditLogger,
    ):
        mock_audit = AsyncMock()
        mock_audit.record = capture_actor
        MockAuditLogger.return_value = mock_audit

        await OverrideExpiryJob().run(sf, registry=None)

    assert len(recorded_actors) == 1
    actor = recorded_actors[0]
    assert actor.sub == "system_expiry"
    assert actor.team_id == "system"
