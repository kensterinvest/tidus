"""Tier 5 — Ollama-hosted local LLM escalation.

T5 fires when the T1+T2+T2b verdict is non-confidential AND a topic-keyword
match suggests the prompt may be a topic-bearing confidential that the
cheaper tiers cannot catch. Per findings.md §3, 6/12 IRR-flipped confidentials
are topic-bearing (financial hardship, credential requests, HR complaints)
with no PII entity markers — the exact class T5 exists for.

Production constraint (hardware-requirements.md): T5 requires a GPU. CPU-only
deployments run the Small-Team SKU with T5 disabled (89.2% recall baseline).
Enterprise SKU deployments run T5 on Ollama with a GPU, targeting p95 ≤ 500 ms.

Design:
    * Async httpx client — non-blocking, can run concurrently with subsequent
      requests in the same event loop
    * Grammar-constrained JSON via Ollama's `format=json` — eliminates
      prompt-parsing failures at source
    * Exact-match TTL cache on the raw prompt — identical messages often
      repeat in enterprise traffic (same user, retry, multi-round)
    * Sliding-window per-worker rate limit — prevents a single tenant from
      starving GPU capacity
    * Graceful degradation — `classify()` returns None on failure; the
      classifier translates that to `confidence_warning=True` and keeps
      the pre-T5 verdict
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict, deque
from typing import Any

import httpx

from tidus.classification.models import LLMResult, LLMUnavailableError
from tidus.observability.classification_metrics import (
    T5_CALLS_TOTAL,
    T5_LATENCY_SECONDS,
)

log = logging.getLogger(__name__)

# Frozen rubric — identical to scripts/label_wildchat.py so the LLM's
# classification matches the encoder's training distribution. Copying inline
# rather than importing from scripts/ keeps the runtime package self-contained.
SYSTEM_PROMPT = """\
You are an expert classifier for an enterprise AI routing system. You classify \
each user message across three dimensions. You output ONLY a single JSON object \
with fields: domain, complexity, privacy, rationale. No preamble, no trailing \
commentary.

TAXONOMY

domain — what TYPE of task (not topic):
  chat            = conversational, open-ended, no clear deliverable
  code            = writing/debugging/explaining code; shell commands; SQL
  reasoning       = logic, math proofs, step-by-step analysis, planning
  extraction      = pulling structured data from unstructured input
  classification  = assigning labels/categories to input
  summarization   = condensing longer input into shorter output
  creative        = fiction, poetry, brainstorming, roleplay, marketing copy

complexity — cognitive load needed for a correct answer:
  simple    = one-step lookup/answer; trivially verifiable
  moderate  = multi-step but bounded scope
  complex   = architecture, system design, advanced reasoning
  critical  = medical diagnosis, legal advice, financial planning, compliance —
              wrong answer has material real-world consequences

privacy — sensitivity of the content itself:
  public        = no sensitive info; could be posted on a public forum
  internal      = business content, work tasks, routine questions
  confidential  = contains PII (SSN, credit cards, real names+context),
                  secrets (API keys, passwords, tokens),
                  medical/legal/financial specifics tied to a person or org

RULES

- Classify the REQUEST the user is making, not the topic they mention.
  "Summarize this Python tutorial" -> domain=summarization, not code.
- When privacy is ambiguous between internal and confidential, choose
  confidential. False negatives here are compliance incidents; overclassification
  is safe. This is asymmetric cost.
- Never output privacy=public if the message contains real names, addresses,
  phone numbers, emails, account numbers, or any identifier.
- rationale = ONE sentence explaining the domain+complexity choice.

OUTPUT FORMAT (valid JSON, nothing else)

{"domain": "...", "complexity": "...", "privacy": "...", "rationale": "..."}
"""

USER_PROMPT_TEMPLATE = """\
Classify this message:

<<<
{message}
>>>

Respond with JSON only."""

MAX_CHARS = 1200

# Voter strength for T5's vote in `TaskClassifier._merge_privacy` — NOT a
# calibrated probability. Ollama's /api/chat does not expose per-token logprobs
# over our taxonomy, so a true probability is unobtainable from this call path.
#
# This value controls how much T5 contributes when it agrees with other voters
# (it's the `base` in `max(base + 0.05 * (n_votes - 1), 1.0)`) and what
# confidence appears on downstream T5-only flips. The chosen 0.95 treats T5 as
# a strong asymmetric-safety voter, equal to a T1 regex hit — appropriate
# because both are high-precision signals in context (T5 only fires on the
# topic-bearing miss class per `_should_escalate_to_t5`).
#
# If you ever replace this with a measured value, measure it as per-vote
# agreement accuracy on IRR-adjudicated ground truth, not as a token
# probability. The merge arithmetic assumes voter-strength semantics.
#
# See backlog task #50 and advisor A.5 review.
LLM_VOTER_STRENGTH = 0.95


def _normalize_model_name(name: str) -> str:
    """Treat `foo` and `foo:latest` as equivalent — Ollama's `/api/tags`
    canonicalizes unqualified pulls to `:latest`."""
    if name.endswith(":latest"):
        return name[: -len(":latest")]
    return name


class _SlidingWindowLimiter:
    """Per-minute sliding-window rate limiter. Thread-safe via asyncio.Lock."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._hits: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        if self._max <= 0:
            return False
        async with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            while self._hits and self._hits[0] < cutoff:
                self._hits.popleft()
            if len(self._hits) >= self._max:
                return False
            self._hits.append(now)
            return True


class _TTLCache:
    """Minimal TTL + LRU cache. Not thread-safe — T5 callers serialize via
    asyncio event-loop ordering (single event loop per worker)."""

    def __init__(self, max_entries: int, ttl_seconds: int) -> None:
        self._max = max_entries
        self._ttl = ttl_seconds
        self._items: OrderedDict[str, tuple[LLMResult, float]] = OrderedDict()

    def get(self, key: str) -> LLMResult | None:
        item = self._items.get(key)
        if item is None:
            return None
        value, expiry = item
        if time.monotonic() > expiry:
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: str, value: LLMResult) -> None:
        expiry = time.monotonic() + self._ttl
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = (value, expiry)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


class LLMClassifier:
    """Ollama-hosted LLM classifier for Tier 5 escalation."""

    def __init__(
        self,
        model: str,
        endpoint: str = "http://localhost:11434",
        rate_limit_per_minute: int = 60,
        cache_ttl_seconds: int = 3600,
        cache_max_entries: int = 10_000,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        self._model = model
        self._endpoint = endpoint.rstrip("/")
        self._limiter = _SlidingWindowLimiter(rate_limit_per_minute)
        self._cache = _TTLCache(cache_max_entries, cache_ttl_seconds)
        self._timeout = request_timeout_seconds
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def cache_size(self) -> int:
        """Exposed for tests + observability."""
        return len(self._cache)

    async def startup(self) -> None:
        """Ping Ollama and verify the configured model is pulled.

        Model-name matching treats `foo` and `foo:latest` as equivalent —
        Ollama canonicalizes to `foo:latest` in `/api/tags` output when the
        default tag is pulled without qualification (advisor A.4 Bug #1).
        Raises LLMUnavailableError on any failure — TaskClassifier catches
        this and disables T5 for the session while keeping T1+T2+T2b live.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._endpoint}/api/tags")
                resp.raise_for_status()
                body = resp.json()
        except Exception as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self._endpoint}: {exc}",
            ) from exc

        available = {_normalize_model_name(m["name"]) for m in body.get("models", [])}
        if _normalize_model_name(self._model) not in available:
            raise LLMUnavailableError(
                f"Ollama model '{self._model}' not pulled. Run: ollama pull {self._model}",
            )
        self._loaded = True
        log.info("T5 LLMClassifier ready: model=%s endpoint=%s", self._model, self._endpoint)

    async def classify(self, text: str) -> LLMResult | None:
        """Classify `text` via Ollama. Returns None on any failure —
        callers treat None as "T5 unavailable, proceed with confidence_warning."

        This method NEVER raises. Failure modes that return None:
          * Rate limit exceeded
          * Ollama unreachable / 5xx
          * Timeout
          * JSON parse failure
          * Schema mismatch (unexpected label values)
        Each failure emits a log line but doesn't propagate.

        Emits Prometheus counters on every exit and a latency histogram for
        the cache-miss path (see `tidus/observability/classification_metrics`).
        Cache hits are not timed — the network call is the dominant cost and
        what we actually want to observe under load.
        """
        if not self._loaded:
            return None

        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            T5_CALLS_TOTAL.labels(result="cache_hit").inc()
            return cached

        if not await self._limiter.try_acquire():
            log.warning("T5 rate limit exceeded; declining classification")
            T5_CALLS_TOTAL.labels(result="rate_limited").inc()
            return None

        start = time.perf_counter()
        try:
            raw = await self._call_ollama(text)
        except Exception as exc:  # noqa: BLE001 — httpx raises many types
            log.warning("T5 Ollama call failed: %s", exc)
            T5_CALLS_TOTAL.labels(result="failure").inc()
            T5_LATENCY_SECONDS.labels(result="failure").observe(time.perf_counter() - start)
            return None

        result = self._parse(raw)
        elapsed = time.perf_counter() - start
        if result is None:
            T5_CALLS_TOTAL.labels(result="failure").inc()
            T5_LATENCY_SECONDS.labels(result="failure").observe(elapsed)
            return None

        self._cache.put(key, result)
        T5_CALLS_TOTAL.labels(result="success").inc()
        T5_LATENCY_SECONDS.labels(result="success").observe(elapsed)
        return result

    # ── Internal helpers ────────────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        # Include model name in the hash so a model swap (e.g., phi3.5 →
        # llama3.2:3b) invalidates prior entries. Without this, stale cache
        # hits from the old model leak into the new model's traffic.
        # Advisor A.4 Bug #3.
        raw = f"{self._model}|{text[:MAX_CHARS]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _call_ollama(self, text: str) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(message=text[:MAX_CHARS])},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 120},
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._endpoint}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _parse(raw: dict[str, Any]) -> LLMResult | None:
        content = raw.get("message", {}).get("content", "")
        if not content:
            log.warning("T5 response had empty content")
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            log.warning("T5 response was not valid JSON: %s", exc)
            return None

        for field in ("domain", "complexity", "privacy"):
            if field not in parsed:
                log.warning("T5 response missing field %s: %r", field, parsed)
                return None

        try:
            return LLMResult(
                domain=parsed["domain"],
                complexity=parsed["complexity"],
                privacy=parsed["privacy"],
                confidence={
                    "domain":     LLM_VOTER_STRENGTH,
                    "complexity": LLM_VOTER_STRENGTH,
                    "privacy":    LLM_VOTER_STRENGTH,
                },
                rationale=parsed.get("rationale"),
            )
        except Exception as exc:  # noqa: BLE001 — Pydantic raises ValidationError
            log.warning("T5 response schema mismatch: %s", exc)
            return None
