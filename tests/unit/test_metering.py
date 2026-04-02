"""Unit tests for MeteringService and resolve_caller_id."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tidus.metering.service import (
    MeteringService,
    MeteringStage,
    resolve_caller_id,
    _YELLOW_THRESHOLD,
    _ORANGE_THRESHOLD,
    _RED_THRESHOLD,
)


# ── resolve_caller_id ──────────────────────────────────────────────────────────

class TestResolveCallerId:
    def test_header_takes_priority(self):
        caller_id, source = resolve_caller_id(
            user_id_header="alice",
            api_key_sub="svc-account",
            client_ip="10.0.0.1",
            user_agent="python-httpx/0.27",
        )
        assert caller_id == "alice"
        assert source == "header"

    def test_api_key_sub_second_priority(self):
        caller_id, source = resolve_caller_id(
            user_id_header=None,
            api_key_sub="svc-account",
            client_ip="10.0.0.1",
            user_agent="python-httpx/0.27",
        )
        assert caller_id == "svc-account"
        assert source == "api_key"

    def test_dev_sub_falls_through_to_ip_hash(self):
        """Dev-mode 'dev' sub should not be used as a caller identity."""
        caller_id, source = resolve_caller_id(
            user_id_header=None,
            api_key_sub="dev",
            client_ip="10.0.0.1",
            user_agent="curl/8.0",
        )
        assert source == "ip_hash"
        assert len(caller_id) == 32  # truncated SHA-256 hex

    def test_ip_hash_fallback(self):
        caller_id, source = resolve_caller_id(
            user_id_header=None,
            api_key_sub=None,
            client_ip="203.0.113.1",
            user_agent="Mozilla/5.0",
        )
        assert source == "ip_hash"
        assert len(caller_id) == 32

    def test_ip_hash_deterministic(self):
        """Same IP + UA should always produce the same hash."""
        a, _ = resolve_caller_id(None, None, "1.2.3.4", "agent/1")
        b, _ = resolve_caller_id(None, None, "1.2.3.4", "agent/1")
        assert a == b

    def test_ip_hash_differs_for_different_inputs(self):
        a, _ = resolve_caller_id(None, None, "1.2.3.4", "agent/1")
        b, _ = resolve_caller_id(None, None, "5.6.7.8", "agent/1")
        assert a != b

    def test_header_whitespace_stripped(self):
        caller_id, source = resolve_caller_id(
            user_id_header="  bob  ",
            api_key_sub=None,
            client_ip=None,
            user_agent=None,
        )
        assert caller_id == "bob"
        assert source == "header"

    def test_missing_all_inputs_still_returns_hash(self):
        caller_id, source = resolve_caller_id(None, None, None, None)
        assert source == "ip_hash"
        assert caller_id  # non-empty


# ── MeteringService (with mocked DB) ─────────────────────────────────────────

class TestMeteringStageLogic:
    """Test stage derivation from active user counts (no DB needed)."""

    def _stage_for(self, count: int) -> MeteringStage:
        """Replicate the stage logic from MeteringService.get_status()."""
        if count >= _RED_THRESHOLD:
            return MeteringStage.threshold
        elif count >= _ORANGE_THRESHOLD:
            return MeteringStage.orange
        elif count >= _YELLOW_THRESHOLD:
            return MeteringStage.yellow
        return MeteringStage.normal

    def test_below_yellow(self):
        assert self._stage_for(0) == MeteringStage.normal
        assert self._stage_for(799) == MeteringStage.normal

    def test_yellow_boundary(self):
        assert self._stage_for(800) == MeteringStage.yellow
        assert self._stage_for(949) == MeteringStage.yellow

    def test_orange_boundary(self):
        assert self._stage_for(950) == MeteringStage.orange
        assert self._stage_for(999) == MeteringStage.orange

    def test_threshold_boundary(self):
        assert self._stage_for(1000) == MeteringStage.threshold
        assert self._stage_for(5000) == MeteringStage.threshold

    def test_thresholds_are_spec_values(self):
        assert _YELLOW_THRESHOLD == 800
        assert _ORANGE_THRESHOLD == 950
        assert _RED_THRESHOLD == 1_000


class TestMeteringServiceWithMockDB:
    """Test MeteringService behaviour using a mocked session factory."""

    def _make_service(self, active_count: int, trend: list[int] | None = None) -> MeteringService:
        mock_sf = MagicMock()
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        # Mock scalar_one() returns for count queries
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = active_count
        mock_session.execute = AsyncMock(return_value=mock_result)

        svc = MeteringService(mock_sf)
        return svc

    @pytest.mark.asyncio
    async def test_record_event_calls_db(self):
        mock_sf = MagicMock()
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        svc = MeteringService(mock_sf)
        await svc.record_event("alice", "header", team_id="eng", path="/api/v1/route")

        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_record_event_is_nonfatal_on_db_error(self):
        mock_sf = MagicMock()
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_session.add = MagicMock(side_effect=RuntimeError("DB down"))

        svc = MeteringService(mock_sf)
        # Should not raise
        await svc.record_event("alice", "header")

    @pytest.mark.asyncio
    async def test_get_status_normal(self):
        svc = self._make_service(active_count=100)
        status = await svc.get_status()
        assert status.active_user_count == 100
        assert status.stage == MeteringStage.normal

    @pytest.mark.asyncio
    async def test_get_status_yellow(self):
        svc = self._make_service(active_count=850)
        status = await svc.get_status()
        assert status.stage == MeteringStage.yellow

    @pytest.mark.asyncio
    async def test_get_status_orange(self):
        svc = self._make_service(active_count=975)
        status = await svc.get_status()
        assert status.stage == MeteringStage.orange

    @pytest.mark.asyncio
    async def test_get_status_threshold(self):
        svc = self._make_service(active_count=1200)
        status = await svc.get_status()
        assert status.stage == MeteringStage.threshold

    @pytest.mark.asyncio
    async def test_to_dict_includes_required_keys(self):
        svc = self._make_service(active_count=500)
        status = await svc.get_status()
        d = status.to_dict()
        assert "active_user_count" in d
        assert "threshold" in d
        assert "stage" in d
        assert "window_start" in d
        assert "window_end" in d
        assert "trend_7d" in d
        assert d["threshold"] == 1000
