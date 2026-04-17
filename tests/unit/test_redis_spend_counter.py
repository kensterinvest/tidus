"""Unit tests for RedisSpendCounter (Fix 10).

Uses fakeredis for CI-safe Redis emulation — no external Redis required.
Tests mirror the in-memory SpendCounter contract so the two backends are
provably interchangeable.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from tidus.cost.counter import RedisSpendCounter


@pytest_asyncio.fixture
async def redis_client():
    """Fresh fakeredis async client per test; flushed on teardown."""
    import fakeredis.aioredis
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


@pytest_asyncio.fixture
async def counter(redis_client):
    return RedisSpendCounter(redis_client, prefix="tidus:test")


# ── Basic contract ───────────────────────────────────────────────────────────

class TestAddGet:
    async def test_add_returns_new_total(self, counter):
        total = await counter.add("team-a", None, 5.0)
        assert total == pytest.approx(5.0)

    async def test_add_accumulates(self, counter):
        await counter.add("team-a", None, 1.0)
        await counter.add("team-a", None, 2.5)
        assert await counter.get("team-a", None) == pytest.approx(3.5)

    async def test_get_missing_scope_returns_zero(self, counter):
        assert await counter.get("unknown-team", None) == 0.0

    async def test_different_scopes_are_independent(self, counter):
        await counter.add("team-a", None, 10.0)
        await counter.add("team-b", None, 20.0)
        assert await counter.get("team-a", None) == pytest.approx(10.0)
        assert await counter.get("team-b", None) == pytest.approx(20.0)

    async def test_workflow_and_team_are_independent_keys(self, counter):
        await counter.add("team-a", None, 10.0)
        await counter.add("team-a", "wf-1", 3.0)
        assert await counter.get("team-a", None) == pytest.approx(10.0)
        assert await counter.get("team-a", "wf-1") == pytest.approx(3.0)


# ── Atomic check_and_add ─────────────────────────────────────────────────────

class TestCheckAndAdd:
    async def test_passes_when_within_limit(self, counter):
        allowed, total = await counter.check_and_add("team-a", None, 5.0, 10.0)
        assert allowed is True
        assert total == pytest.approx(5.0)

    async def test_rejects_when_would_exceed(self, counter):
        await counter.add("team-a", None, 8.0)
        allowed, current = await counter.check_and_add("team-a", None, 5.0, 10.0)
        assert allowed is False
        assert current == pytest.approx(8.0)
        # No partial deduction on reject
        assert await counter.get("team-a", None) == pytest.approx(8.0)

    async def test_exact_boundary_allowed(self, counter):
        allowed, total = await counter.check_and_add("team-a", None, 10.0, 10.0)
        assert allowed is True
        assert total == pytest.approx(10.0)

    async def test_concurrent_check_and_add_bounded_by_limit(self, counter):
        """20 concurrent check_and_add($0.10) on a $1.00 limit — exactly 10 pass.

        This is the core correctness guarantee that was the whole reason to
        move to Redis — Lua script atomicity prevents cross-worker races.
        """
        limit = 1.0
        amount = 0.10
        results = await asyncio.gather(*[
            counter.check_and_add("team-race", None, amount, limit)
            for _ in range(20)
        ])
        passed = sum(1 for allowed, _ in results if allowed)
        assert passed == 10, f"Expected exactly 10 to fit $1.00 limit, got {passed}"
        final_total = await counter.get("team-race", None)
        assert final_total <= limit + 1e-9


# ── Reset ────────────────────────────────────────────────────────────────────

class TestReset:
    async def test_reset_zeros_scope(self, counter):
        await counter.add("team-a", None, 5.0)
        await counter.reset("team-a", None)
        assert await counter.get("team-a", None) == 0.0

    async def test_reset_leaves_other_scopes_alone(self, counter):
        await counter.add("team-a", None, 5.0)
        await counter.add("team-b", None, 7.0)
        await counter.reset("team-a", None)
        assert await counter.get("team-b", None) == pytest.approx(7.0)

    async def test_reset_workflow_clears_all_teams_for_that_workflow(self, counter):
        await counter.add("team-a", "wf-shared", 1.0)
        await counter.add("team-b", "wf-shared", 2.0)
        await counter.add("team-c", "wf-other", 5.0)
        await counter.add("team-a", None, 9.0)  # team-level counter

        deleted = await counter.reset_workflow("wf-shared")
        assert deleted == 2
        assert await counter.get("team-a", "wf-shared") == 0.0
        assert await counter.get("team-b", "wf-shared") == 0.0
        # Other workflow and team counters untouched
        assert await counter.get("team-c", "wf-other") == pytest.approx(5.0)
        assert await counter.get("team-a", None) == pytest.approx(9.0)


# ── get_all ──────────────────────────────────────────────────────────────────

class TestGetAll:
    async def test_snapshot_returns_all_counters(self, counter):
        await counter.add("team-a", None, 1.0)
        await counter.add("team-b", None, 2.0)
        await counter.add("team-a", "wf-1", 0.5)
        snap = await counter.get_all()
        assert snap[("team-a", None)] == pytest.approx(1.0)
        assert snap[("team-b", None)] == pytest.approx(2.0)
        assert snap[("team-a", "wf-1")] == pytest.approx(0.5)

    async def test_empty_when_no_counters(self, counter):
        snap = await counter.get_all()
        assert snap == {}


# ── Multi-worker (shared Redis) ──────────────────────────────────────────────

class TestCrossWorkerConsistency:
    """The whole point of moving to Redis: two worker processes see the same state."""

    async def test_two_counters_sharing_redis_see_same_state(self, redis_client):
        worker_a = RedisSpendCounter(redis_client, prefix="tidus:test")
        worker_b = RedisSpendCounter(redis_client, prefix="tidus:test")

        await worker_a.add("team-eng", None, 3.0)
        # Worker B sees worker A's spend
        assert await worker_b.get("team-eng", None) == pytest.approx(3.0)

        # Interleaved writes from both workers accumulate correctly
        await worker_b.add("team-eng", None, 1.5)
        await worker_a.add("team-eng", None, 0.25)
        assert await worker_a.get("team-eng", None) == pytest.approx(4.75)
        assert await worker_b.get("team-eng", None) == pytest.approx(4.75)

    async def test_concurrent_workers_cannot_overrun_limit(self, redis_client):
        """20 check_and_add($0.10, limit=$1.00) across 2 simulated workers — 10 pass."""
        worker_a = RedisSpendCounter(redis_client, prefix="tidus:test")
        worker_b = RedisSpendCounter(redis_client, prefix="tidus:test")

        calls = []
        for i in range(20):
            worker = worker_a if i % 2 == 0 else worker_b
            calls.append(worker.check_and_add("team-race", None, 0.10, 1.0))
        results = await asyncio.gather(*calls)

        passed = sum(1 for allowed, _ in results if allowed)
        assert passed == 10, (
            f"Cross-worker race allowed {passed} requests to pass a $1.00 limit "
            f"at $0.10 each — Redis atomicity broken"
        )
        final_total = await worker_a.get("team-race", None)
        assert final_total <= 1.0 + 1e-9
