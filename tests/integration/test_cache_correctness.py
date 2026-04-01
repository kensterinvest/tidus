"""Integration tests for ExactCache correctness.

Verifies that:
- Cache hits return byte-for-byte identical content
- Different teams with the same prompt get different cache keys (no cross-team leakage)
- TTL expiry evicts stale entries
- Cache statistics (hit/miss counters) are accurate
- Key stability: message dicts with re-ordered keys still produce the same hash

Run with:
    uv run pytest tests/integration/test_cache_correctness.py -v
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from tidus.cache.exact_cache import ExactCache


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cache() -> ExactCache:
    return ExactCache(ttl_seconds=3600)


MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]
MODEL_ID = "deepseek-v3"
TEAM_A = "team-engineering"
TEAM_B = "team-marketing"


# ── Cache correctness ─────────────────────────────────────────────────────────

class TestCacheHitsReturnIdenticalContent:
    async def test_first_call_is_miss(self, cache):
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        result = await cache.get(key)
        assert result is None

    async def test_set_then_get_returns_same_content(self, cache):
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        await cache.set(key, "Paris", MODEL_ID)

        hit = await cache.get(key)
        assert hit == "Paris"

    async def test_cached_content_is_byte_identical(self, cache):
        """The exact bytes stored must be returned — no mutation or re-encoding."""
        content = "La réponse est: 42 🌍"
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        await cache.set(key, content, MODEL_ID)

        assert await cache.get(key) == content

    async def test_repeated_gets_all_return_same_content(self, cache):
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        await cache.set(key, "Consistent answer", MODEL_ID)

        results = [await cache.get(key) for _ in range(10)]
        assert all(r == "Consistent answer" for r in results)


# ── Cross-team isolation ──────────────────────────────────────────────────────

class TestCrossTeamCacheIsolation:
    async def test_same_messages_different_teams_produce_different_keys(self, cache):
        """team_id is part of the key — teams must never share a cache entry."""
        key_a = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        key_b = cache.make_key(TEAM_B, MESSAGES, MODEL_ID)
        assert key_a != key_b

    async def test_team_a_response_not_visible_to_team_b(self, cache):
        """Storing a response for team-A must not produce a hit for team-B."""
        key_a = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        await cache.set(key_a, "Team A answer", MODEL_ID)

        key_b = cache.make_key(TEAM_B, MESSAGES, MODEL_ID)
        assert await cache.get(key_b) is None

    async def test_both_teams_can_independently_cache_same_prompt(self, cache):
        """Teams can each cache the same prompt with different responses."""
        key_a = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        key_b = cache.make_key(TEAM_B, MESSAGES, MODEL_ID)

        await cache.set(key_a, "Answer for A", MODEL_ID)
        await cache.set(key_b, "Answer for B", MODEL_ID)

        assert await cache.get(key_a) == "Answer for A"
        assert await cache.get(key_b) == "Answer for B"


# ── Key stability ─────────────────────────────────────────────────────────────

class TestKeyStability:
    def test_dict_key_order_does_not_affect_cache_key(self, cache):
        """Messages with same content but different key order must hash identically."""
        msgs_v1 = [{"role": "user", "content": "hello"}]
        msgs_v2 = [{"content": "hello", "role": "user"}]  # keys reversed

        assert cache.make_key(TEAM_A, msgs_v1, MODEL_ID) == \
               cache.make_key(TEAM_A, msgs_v2, MODEL_ID)

    def test_different_model_ids_produce_different_keys(self, cache):
        key_gpt = cache.make_key(TEAM_A, MESSAGES, "gpt-4o")
        key_claude = cache.make_key(TEAM_A, MESSAGES, "claude-sonnet-4-5")
        assert key_gpt != key_claude

    def test_different_messages_produce_different_keys(self, cache):
        msgs_a = [{"role": "user", "content": "hello"}]
        msgs_b = [{"role": "user", "content": "goodbye"}]
        assert cache.make_key(TEAM_A, msgs_a, MODEL_ID) != \
               cache.make_key(TEAM_A, msgs_b, MODEL_ID)

    def test_key_is_deterministic_across_calls(self, cache):
        k1 = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        k2 = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        assert k1 == k2


# ── TTL eviction ──────────────────────────────────────────────────────────────

class TestTTLEviction:
    async def test_entry_evicted_after_ttl_expires(self):
        """An entry stored with ttl=1 must be a miss after the clock advances."""
        cache = ExactCache(ttl_seconds=1)
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        await cache.set(key, "Ephemeral answer", MODEL_ID)

        # Advance the monotonic clock by 2 seconds
        future_time = time.monotonic() + 2
        with patch("tidus.cache.exact_cache.time") as mock_time:
            mock_time.monotonic.return_value = future_time
            result = await cache.get(key)

        assert result is None

    async def test_entry_still_valid_before_ttl_expires(self):
        """An entry within its TTL window must still be returned."""
        cache = ExactCache(ttl_seconds=3600)
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        await cache.set(key, "Valid answer", MODEL_ID)

        # Only 10 seconds passed — well within TTL
        future_time = time.monotonic() + 10
        with patch("tidus.cache.exact_cache.time") as mock_time:
            mock_time.monotonic.return_value = future_time
            result = await cache.get(key)

        assert result == "Valid answer"

    async def test_expired_entry_is_removed_from_store(self):
        """get() on an expired key must physically remove it from the internal store."""
        cache = ExactCache(ttl_seconds=1)
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        await cache.set(key, "Gone soon", MODEL_ID)
        assert cache.stats()["size"] == 1

        future_time = time.monotonic() + 2
        with patch("tidus.cache.exact_cache.time") as mock_time:
            mock_time.monotonic.return_value = future_time
            await cache.get(key)

        assert cache.stats()["size"] == 0


# ── Statistics accuracy ───────────────────────────────────────────────────────

class TestCacheStatistics:
    async def test_miss_counter_increments_on_cold_get(self, cache):
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        await cache.get(key)  # miss
        assert cache.stats()["misses"] == 1

    async def test_hit_counter_increments_after_set(self, cache):
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        await cache.set(key, "answer", MODEL_ID)
        await cache.get(key)  # hit
        assert cache.stats()["hits"] == 1

    async def test_hit_rate_is_accurate_after_mixed_calls(self, cache):
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)

        await cache.get(key)           # miss
        await cache.get(key)           # miss
        await cache.set(key, "x", MODEL_ID)
        await cache.get(key)           # hit
        await cache.get(key)           # hit

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate_pct"] == pytest.approx(50.0)

    async def test_invalidate_removes_entry(self, cache):
        key = cache.make_key(TEAM_A, MESSAGES, MODEL_ID)
        await cache.set(key, "will be deleted", MODEL_ID)
        await cache.invalidate(key)

        assert await cache.get(key) is None
        assert cache.stats()["size"] == 0
