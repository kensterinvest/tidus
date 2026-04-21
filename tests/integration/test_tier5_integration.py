"""Integration test — real Ollama T5 with Phi-3.5-mini.

Slow. Each call is ~15-40s on CPU (our bench), ~200-500ms on GPU.
Skipped automatically when Ollama isn't reachable or the model isn't pulled.
Run locally with: ollama serve (separate terminal), then:
    uv run pytest tests/integration/test_tier5_integration.py -v
"""
from __future__ import annotations

import httpx
import pytest

from tidus.classification import TaskClassifier
from tidus.classification.llm_classifier import LLMClassifier
from tidus.classification.models import LLMUnavailableError

MODEL = "phi3.5:3.8b-mini-instruct-q4_K_M"
ENDPOINT = "http://localhost:11434"


def _ollama_ready() -> tuple[bool, str]:
    try:
        r = httpx.get(f"{ENDPOINT}/api/tags", timeout=3.0)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if MODEL not in models:
            return False, f"model {MODEL} not pulled"
        return True, "ok"
    except Exception as exc:
        return False, f"ollama unreachable: {exc}"


_READY, _SKIP_REASON = _ollama_ready()

needs_ollama = pytest.mark.skipif(
    not _READY,
    reason=f"T5 integration requires live Ollama: {_SKIP_REASON}. "
           f"Start with `ollama serve` and `ollama pull {MODEL}`.",
)


@needs_ollama
class TestLLMClassifierReal:
    @pytest.mark.asyncio
    async def test_startup_ok(self):
        client = LLMClassifier(model=MODEL, endpoint=ENDPOINT)
        await client.startup()
        assert client.loaded

    @pytest.mark.asyncio
    async def test_real_classification_returns_valid_result(self):
        """Send a classifiable prompt through real Phi-3.5-mini."""
        client = LLMClassifier(model=MODEL, endpoint=ENDPOINT, request_timeout_seconds=120.0)
        await client.startup()
        r = await client.classify(
            "I'm having a mental health crisis and need help finding a therapist.",
        )
        assert r is not None
        # The encoder+rubric guidance leans this toward confidential/critical
        # but we don't strictly assert classification correctness — only that
        # the response parses and fits the taxonomy.
        assert r.domain in {
            "chat", "code", "reasoning", "extraction",
            "classification", "summarization", "creative",
        }
        assert r.complexity in {"simple", "moderate", "complex", "critical"}
        assert r.privacy in {"public", "internal", "confidential"}
        assert r.rationale is not None and len(r.rationale) > 0

    @pytest.mark.asyncio
    async def test_cache_hit_on_repeat(self):
        client = LLMClassifier(model=MODEL, endpoint=ENDPOINT, request_timeout_seconds=120.0)
        await client.startup()
        before = client.cache_size
        r1 = await client.classify("what's the forecast for next week")
        r2 = await client.classify("what's the forecast for next week")
        assert r1 == r2  # identical prompts → identical cached result
        assert client.cache_size == before + 1  # only one entry added


@needs_ollama
class TestTaskClassifierWithRealT5:
    @pytest.mark.asyncio
    async def test_cascade_escalates_to_t5_on_hardship_prompt(self):
        """Hardship keyword triggers T5; LLM may or may not flip to confidential."""
        llm = LLMClassifier(model=MODEL, endpoint=ENDPOINT, request_timeout_seconds=120.0)
        await llm.startup()
        clf = TaskClassifier(llm=llm)  # no encoder — isolate T5 path via T1-only baseline
        r = await clf.classify_async(
            "I can't afford my rent this month, I might get evicted.",
            include_debug=True,
        )
        # T5 fired (hardship keyword). Whether the LLM says confidential or
        # internal is the LLM's call; assert only that T5 was consulted.
        assert r.debug is not None
        assert r.debug.get("tier5_llm") is not None


class TestNoOllama:
    """Runs always — verifies graceful degradation path without Ollama."""

    @pytest.mark.asyncio
    async def test_startup_raises_llm_unavailable_when_no_server(self):
        client = LLMClassifier(model="fake:model", endpoint="http://127.0.0.1:1")
        with pytest.raises(LLMUnavailableError):
            await client.startup()

    @pytest.mark.asyncio
    async def test_classifier_startup_degrades_gracefully(self):
        from tidus.settings import Settings
        settings = Settings()
        settings.classify_tier5_enabled = True
        # Point at an unreachable endpoint
        settings.ollama_base_url = "http://127.0.0.1:1"
        settings.classify_tier5_model = "fake:nonexistent"

        clf = TaskClassifier(settings=settings)
        await clf.startup()  # must not raise
        # After failed T5 startup, attempting classify_async should still succeed
        r = await clf.classify_async("hello")
        assert r is not None
        # LLM was never loaded; no confidence_warning (T5 wasn't attempted)
        assert r.confidence_warning is False
