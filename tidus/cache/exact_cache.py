"""Exact-match response cache — Pillar 3, Layer 1.

Caches responses by SHA-256(team_id + json(messages) + model_id).
If the same team sends the exact same messages to the same model,
the stored response is returned immediately at zero vendor cost.

Cache keys are always team-scoped. A cached response from team-A is
never returned to team-B, even for identical prompts.
Confidential-privacy tasks are never cached (privacy guard in /complete).

Backend: in-memory dict (dev) → Redis (production via REDIS_URL).

Example:
    cache = ExactCache(ttl_seconds=3600)
    key = cache.make_key("team-eng", messages, "deepseek-v3")
    hit = await cache.get(key)
    if hit is None:
        response = await adapter.complete(...)
        await cache.set(key, response.content)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass


@dataclass
class CacheEntry:
    content: str
    model_id: str
    stored_at: float  # monotonic time
    ttl_seconds: int


class ExactCache:
    """In-memory exact-match cache with TTL eviction."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._ttl = ttl_seconds
        self.hits: int = 0
        self.misses: int = 0

    def make_key(self, team_id: str, messages: list[dict], model_id: str) -> str:
        """SHA-256 hash of team_id + canonicalised messages + model_id."""
        payload = json.dumps(
            {"team_id": team_id, "messages": messages, "model_id": model_id},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def get(self, key: str) -> str | None:
        """Return cached content or None on miss/expiry."""
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if time.monotonic() - entry.stored_at > entry.ttl_seconds:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return entry.content

    async def set(self, key: str, content: str, model_id: str) -> None:
        """Store a response."""
        self._store[key] = CacheEntry(
            content=content,
            model_id=model_id,
            stored_at=time.monotonic(),
            ttl_seconds=self._ttl,
        )

    async def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(self.hits / total * 100, 1) if total else 0.0,
        }
