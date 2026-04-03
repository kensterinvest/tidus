"""Semantic response cache — Pillar 3, Layer 2.

Catches "same question, different wording" by embedding queries and
finding nearest stored responses above a cosine similarity threshold.

Embedding backend: sentence-transformers (local, no API cost) with
`all-MiniLM-L6-v2` as the default model (~80MB, fast on CPU).

Falls back gracefully to a no-op if sentence-transformers is not installed
so the rest of the system works without the optional dependency.

Example:
    cache = SemanticCache(threshold=0.95, ttl_seconds=900)
    hit = await cache.get("team-eng", messages)
    if hit is None:
        response = await adapter.complete(...)
        await cache.set("team-eng", messages, response.content, model_id)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

_embedder = None
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            _embedder = SentenceTransformer(_EMBEDDING_MODEL)
        except ImportError:
            log.warning(
                "semantic_cache_disabled",
                reason="sentence-transformers not installed",
                install="pip install sentence-transformers",
            )
            _embedder = False  # sentinel: don't retry
    return _embedder if _embedder is not False else None


def _embed(text: str) -> list[float] | None:
    embedder = _get_embedder()
    if embedder is None:
        return None
    try:
        return embedder.encode(text, normalize_embeddings=True).tolist()
    except Exception:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    # Vectors are pre-normalised so magnitude = 1
    return dot


def _messages_to_text(messages: list[dict]) -> str:
    return " ".join(
        m.get("content", "") for m in messages
        if isinstance(m.get("content"), str)
    )


@dataclass
class SemanticEntry:
    team_id: str
    text: str
    embedding: list[float]
    content: str
    model_id: str
    stored_at: float
    ttl_seconds: int


class SemanticCache:
    """In-memory semantic cache using cosine similarity over embeddings."""

    def __init__(self, threshold: float = 0.95, ttl_seconds: int = 900) -> None:
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._entries: list[SemanticEntry] = []
        self.hits: int = 0
        self.misses: int = 0

    async def get(self, team_id: str, messages: list[dict]) -> str | None:
        """Return cached content if a similar query exists above the threshold."""
        text = _messages_to_text(messages)
        query_emb = _embed(text)
        if query_emb is None:
            self.misses += 1
            return None

        now = time.monotonic()
        best_score = 0.0
        best_entry = None

        for entry in self._entries:
            if entry.team_id != team_id:
                continue
            if now - entry.stored_at > entry.ttl_seconds:
                continue
            score = _cosine_similarity(query_emb, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self._threshold:
            self.hits += 1
            log.debug("semantic_cache_hit", score=round(best_score, 4), team_id=team_id)
            return best_entry.content

        self.misses += 1
        return None

    async def set(
        self,
        team_id: str,
        messages: list[dict],
        content: str,
        model_id: str,
    ) -> None:
        """Store a response with its embedding."""
        text = _messages_to_text(messages)
        emb = _embed(text)
        if emb is None:
            return
        self._entries.append(
            SemanticEntry(
                team_id=team_id,
                text=text,
                embedding=emb,
                content=content,
                model_id=model_id,
                stored_at=time.monotonic(),
                ttl_seconds=self._ttl,
            )
        )
        # Prune expired entries to keep memory bounded
        now = time.monotonic()
        self._entries = [
            e for e in self._entries
            if now - e.stored_at <= e.ttl_seconds
        ]

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(self.hits / total * 100, 1) if total else 0.0,
            "threshold": self._threshold,
        }
