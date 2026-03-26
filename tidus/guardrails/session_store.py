"""In-memory agent session tracker.

Tracks active multi-agent sessions: depth, retry count, and total tokens
consumed. The interface is designed for a Redis swap-in — replace the dict
with Redis HSET/HINCRBY and the rest of the system is unchanged.

Example:
    store = SessionStore()
    session = await store.create("agent-session-abc", max_depth=5)
    updated = await store.increment_depth("agent-session-abc")
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from tidus.models.guardrails import AgentSession, GuardrailPolicy


class SessionStore:
    """Thread-safe in-process store for active agent sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        session_id: str,
        policy: GuardrailPolicy,
        team_id: str = "",
    ) -> AgentSession:
        """Create and store a new agent session.

        Raises ValueError if a session with the same ID already exists.

        Example:
            session = await store.create("sess-123", policy)
        """
        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id!r} already exists")
            session = AgentSession(
                session_id=session_id,
                team_id=team_id,
                max_depth=policy.max_agent_depth,
                max_tokens_per_step=policy.max_tokens_per_step,
                max_retries=policy.max_retries_per_task,
                current_depth=0,
                retry_count=0,
                total_tokens_used=0,
                started_at=datetime.now(UTC),
            )
            self._sessions[session_id] = session
            return session

    async def get(self, session_id: str) -> AgentSession | None:
        """Return the session or None if it doesn't exist."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def increment_depth(self, session_id: str) -> AgentSession | None:
        """Increment agent_depth by 1. Returns updated session or None if not found."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            updated = session.model_copy(update={"current_depth": session.current_depth + 1})
            self._sessions[session_id] = updated
            return updated

    async def increment_retries(self, session_id: str) -> AgentSession | None:
        """Increment retry_count by 1. Returns updated session or None if not found."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            updated = session.model_copy(update={"retry_count": session.retry_count + 1})
            self._sessions[session_id] = updated
            return updated

    async def add_tokens(self, session_id: str, tokens: int) -> AgentSession | None:
        """Add to total_tokens_used. Returns updated session or None if not found."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            updated = session.model_copy(
                update={"total_tokens_used": session.total_tokens_used + tokens}
            )
            self._sessions[session_id] = updated
            return updated

    async def terminate(self, session_id: str) -> bool:
        """Remove a session. Returns True if it existed, False if not found."""
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def list_active(self) -> list[AgentSession]:
        """Return all active sessions (snapshot)."""
        async with self._lock:
            return list(self._sessions.values())
