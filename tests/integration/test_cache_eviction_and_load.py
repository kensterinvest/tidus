"""Integration tests for cache eviction, max-size cap, and concurrent access.

Validates the CRIT-4 fix (unbounded cache growth) and cache correctness
under concurrent write/read patterns.

Key invariants:
- Cache size never exceeds max_size
- Oldest entries are evicted (not random or newest)
- TTL expiry is respected under concurrent access
- Stats remain consistent under concurrent load
- Expired entries don't linger when set() evicts

Run with:
    uv run pytest tests/integration/test_cache_eviction_and_load.py -v
"""

from __future__ import annotations

import asyncio
import time

from tidus.cache.exact_cache import ExactCache

# ── Max-size eviction ─────────────────────────────────────────────────────────

class TestMaxSizeEviction:
    async def test_cache_never_exceeds_max_size(self):
        """Adding entries beyond max_size must not grow the store beyond cap."""
        cache = ExactCache(ttl_seconds=3600, max_size=10)

        for i in range(25):
            await cache.set(f"key-{i}", f"content-{i}", "model-a")

        assert len(cache._store) <= 10

    async def test_oldest_entries_evicted_first(self):
        """When cap is reached, the oldest 10% are removed."""
        cache = ExactCache(ttl_seconds=3600, max_size=10)

        # Fill to capacity
        for i in range(10):
            await cache.set(f"key-{i}", f"content-{i}", "model-a")

        # Adding one more should evict the oldest 10% (1 entry here)
        await cache.set("key-new", "content-new", "model-a")

        # The first key should be evicted (oldest)
        assert await cache.get("key-0") is None
        # Newer entries still present
        assert await cache.get("key-new") == "content-new"

    async def test_eviction_removes_ten_percent(self):
        """On overflow, exactly 10% of entries (rounded up) are removed."""
        cache = ExactCache(ttl_seconds=3600, max_size=100)

        for i in range(100):
            await cache.set(f"key-{i}", f"value-{i}", "model-a")

        # Add one more to trigger eviction of 10 entries
        await cache.set("overflow", "value", "model-a")

        # Store should have 91 entries (100 - 10 + 1)
        assert len(cache._store) == 91

    async def test_size_stays_bounded_under_sustained_writes(self):
        """10,000 writes to a max_size=500 cache must never exceed 500."""
        cache = ExactCache(ttl_seconds=3600, max_size=500)

        for i in range(10_000):
            await cache.set(f"key-{i}", f"v-{i}", "m")
            assert len(cache._store) <= 500, (
                f"Cache exceeded max_size at entry {i}: {len(cache._store)}"
            )


# ── TTL eviction under concurrent access ─────────────────────────────────────

class TestTTLConcurrency:
    async def test_concurrent_gets_on_expired_entry_no_error(self):
        """50 concurrent get() calls on an expired key must not raise."""
        cache = ExactCache(ttl_seconds=1, max_size=1000)
        await cache.set("expiring-key", "data", "model-a")

        # Manually expire the entry
        cache._store["expiring-key"].stored_at = time.monotonic() - 2.0

        results = await asyncio.gather(*[
            cache.get("expiring-key") for _ in range(50)
        ])

        # All must return None (expired), none must raise
        assert all(r is None for r in results)

    async def test_concurrent_writes_same_key_no_corruption(self):
        """20 concurrent set() calls on the same key must not corrupt the store."""
        cache = ExactCache(ttl_seconds=3600, max_size=1000)

        await asyncio.gather(*[
            cache.set("shared-key", f"content-{i}", "model-a")
            for i in range(20)
        ])

        result = await cache.get("shared-key")
        # Some version must be stored — no corruption
        assert result is not None
        assert result.startswith("content-")

    async def test_concurrent_mixed_reads_and_writes(self):
        """Interleaved reads and writes must not produce errors or lost updates."""
        cache = ExactCache(ttl_seconds=3600, max_size=1000)

        async def writer(i):
            await cache.set(f"key-{i}", f"val-{i}", "model")

        async def reader(i):
            return await cache.get(f"key-{i}")

        await asyncio.gather(
            *[writer(i) for i in range(50)],
            *[reader(i) for i in range(50)],
        )

        # After all writers complete, every key must be readable
        for i in range(50):
            val = await cache.get(f"key-{i}")
            assert val == f"val-{i}"


# ── Stats consistency under load ─────────────────────────────────────────────

class TestStatsConsistency:
    async def test_hit_and_miss_counts_consistent_under_load(self):
        """After N concurrent operations, hits + misses == total reads."""
        cache = ExactCache(ttl_seconds=3600, max_size=1000)

        # Pre-seed 25 entries
        for i in range(25):
            await cache.set(f"key-{i}", f"val-{i}", "model")

        # Issue 50 reads: 25 hits + 25 misses
        await asyncio.gather(*[
            cache.get(f"key-{i}") for i in range(50)
        ])

        total = cache.hits + cache.misses
        assert total == 50
        assert cache.hits == 25
        assert cache.misses == 25

    async def test_stats_not_negative_under_any_scenario(self):
        """Hits and misses must never be negative."""
        cache = ExactCache(ttl_seconds=1, max_size=100)

        for i in range(10):
            await cache.set(f"k-{i}", f"v-{i}", "m")

        # Expire all entries
        for entry in cache._store.values():
            entry.stored_at = time.monotonic() - 2.0

        await asyncio.gather(*[cache.get(f"k-{i}") for i in range(10)])

        assert cache.hits >= 0
        assert cache.misses >= 0


# ── Invalidation under concurrency ───────────────────────────────────────────

class TestConcurrentInvalidation:
    async def test_invalidate_while_concurrent_reads(self):
        """invalidate() during concurrent reads must not raise KeyError."""
        cache = ExactCache(ttl_seconds=3600, max_size=1000)
        await cache.set("target", "value", "model")

        async def reader():
            return await cache.get("target")

        async def invalidator():
            await cache.invalidate("target")

        # Run readers and invalidator concurrently
        await asyncio.gather(
            *[reader() for _ in range(20)],
            invalidator(),
        )

        # After invalidation the key must be gone
        assert await cache.get("target") is None

    async def test_double_invalidate_is_safe(self):
        """Calling invalidate() twice on the same key must not raise."""
        cache = ExactCache(ttl_seconds=3600, max_size=100)
        await cache.set("key", "value", "model")
        await cache.invalidate("key")
        await cache.invalidate("key")  # must not raise KeyError
