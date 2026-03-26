"""Unit tests for AgentGuard and SessionStore."""

from __future__ import annotations

import pytest

from tidus.guardrails.agent_guard import AgentGuard
from tidus.guardrails.session_store import SessionStore
from tidus.models.guardrails import GuardrailPolicy


@pytest.fixture
def policy():
    return GuardrailPolicy(max_agent_depth=3, max_tokens_per_step=1000, max_retries_per_task=2)


@pytest.fixture
def store():
    return SessionStore()


@pytest.fixture
def guard(policy, store):
    return AgentGuard(policy, store)


# ── SessionStore ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_and_get_session(store, policy):
    session = await store.create("sess-1", policy, team_id="team-eng")
    assert session.session_id == "sess-1"
    assert session.current_depth == 0

    fetched = await store.get("sess-1")
    assert fetched is not None
    assert fetched.session_id == "sess-1"


@pytest.mark.asyncio
async def test_duplicate_session_raises(store, policy):
    await store.create("sess-dup", policy)
    with pytest.raises(ValueError, match="already exists"):
        await store.create("sess-dup", policy)


@pytest.mark.asyncio
async def test_increment_depth(store, policy):
    await store.create("sess-depth", policy)
    updated = await store.increment_depth("sess-depth")
    assert updated.current_depth == 1
    updated2 = await store.increment_depth("sess-depth")
    assert updated2.current_depth == 2


@pytest.mark.asyncio
async def test_terminate_removes_session(store, policy):
    await store.create("sess-term", policy)
    removed = await store.terminate("sess-term")
    assert removed is True
    assert await store.get("sess-term") is None


@pytest.mark.asyncio
async def test_terminate_nonexistent_returns_false(store):
    assert await store.terminate("no-such-session") is False


# ── AgentGuard ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_guard_allows_first_step(store, guard, policy):
    await store.create("sess-ok", policy)
    result = await guard.check_and_advance("sess-ok", input_tokens=100)
    assert result.allowed is True
    assert result.reason is None


@pytest.mark.asyncio
async def test_guard_rejects_unknown_session(guard):
    result = await guard.check_and_advance("no-session", input_tokens=100)
    assert result.allowed is False
    assert "not found" in result.reason


@pytest.mark.asyncio
async def test_guard_rejects_at_max_depth(store, guard, policy):
    await store.create("sess-deep", policy)
    # Advance to max depth (3 steps: depth becomes 1, 2, 3)
    for _ in range(3):
        result = await guard.check_and_advance("sess-deep", input_tokens=100)
        assert result.allowed is True
    # 4th step should be rejected (would reach depth 4 > max 3)
    result = await guard.check_and_advance("sess-deep", input_tokens=100)
    assert result.allowed is False
    assert "depth" in result.reason.lower()


@pytest.mark.asyncio
async def test_guard_rejects_tokens_per_step_exceeded(store, guard, policy):
    await store.create("sess-tokens", policy)
    result = await guard.check_and_advance("sess-tokens", input_tokens=1001)
    assert result.allowed is False
    assert "tokens" in result.reason.lower()


@pytest.mark.asyncio
async def test_guard_increments_total_tokens(store, guard, policy):
    await store.create("sess-total", policy)
    await guard.check_and_advance("sess-total", input_tokens=300)
    await guard.check_and_advance("sess-total", input_tokens=400)
    session = await store.get("sess-total")
    assert session.total_tokens_used == 700
