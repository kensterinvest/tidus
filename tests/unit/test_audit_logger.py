"""Unit tests for Phase 9 AuditLogger.

Verifies:
- Records are written to the DB on success
- Logger is non-fatal: DB errors are caught and logged, never re-raised
- Fields (actor, action, outcome, metadata) are persisted correctly
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tidus.audit.logger import AuditLogger
from tidus.auth.middleware import TokenPayload


@pytest.fixture
def actor() -> TokenPayload:
    return TokenPayload(
        sub="user-123",
        team_id="team-eng",
        role="developer",
        permissions=[],
        raw_claims={},
    )


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.add = MagicMock()             # sync — SQLAlchemy add() is not async
    session.commit = AsyncMock()
    return session


@pytest.fixture
def mock_factory(mock_session):
    factory = MagicMock()
    factory.return_value = mock_session
    return factory


class TestAuditLoggerRecord:
    async def test_adds_row_and_commits(self, actor, mock_factory, mock_session):
        audit = AuditLogger(session_factory=mock_factory)
        await audit.record(actor=actor, action="route", outcome="success")

        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    async def test_row_has_correct_actor_fields(self, actor, mock_factory, mock_session):
        audit = AuditLogger(session_factory=mock_factory)
        await audit.record(actor=actor, action="complete", resource_id="task-abc")

        row = mock_session.add.call_args[0][0]
        assert row.actor_team_id == "team-eng"
        assert row.actor_role == "developer"
        assert row.actor_sub == "user-123"
        assert row.action == "complete"
        assert row.resource_id == "task-abc"

    async def test_metadata_is_stored(self, actor, mock_factory, mock_session):
        audit = AuditLogger(session_factory=mock_factory)
        await audit.record(
            actor=actor,
            action="complete",
            metadata={"cost_usd": 0.001, "model": "gpt-4o"},
        )
        row = mock_session.add.call_args[0][0]
        assert row.metadata_ == {"cost_usd": 0.001, "model": "gpt-4o"}

    async def test_non_fatal_on_db_error(self, actor, mock_factory, mock_session):
        mock_session.commit = AsyncMock(side_effect=RuntimeError("DB down"))
        audit = AuditLogger(session_factory=mock_factory)

        # Must not raise
        await audit.record(actor=actor, action="route", outcome="error")

    async def test_outcome_rejected_stores_reason(self, actor, mock_factory, mock_session):
        audit = AuditLogger(session_factory=mock_factory)
        await audit.record(
            actor=actor,
            action="route",
            outcome="rejected",
            rejection_reason="budget_exceeded",
        )
        row = mock_session.add.call_args[0][0]
        assert row.outcome == "rejected"
        assert row.rejection_reason == "budget_exceeded"
