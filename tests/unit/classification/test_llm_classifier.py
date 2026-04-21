"""Stage A.4 tests for LLMClassifier (Ollama T5).

Uses respx to mock /api/tags (startup) and /api/chat (classification)
so these tests exercise the real httpx path without a live Ollama server.
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from tidus.classification.llm_classifier import LLMClassifier, _SlidingWindowLimiter
from tidus.classification.models import LLMUnavailableError


def _ollama_chat_response(domain="chat", complexity="moderate", privacy="internal",
                          rationale="test") -> dict:
    """Build a realistic Ollama /api/chat response."""
    content = json.dumps({
        "domain": domain, "complexity": complexity,
        "privacy": privacy, "rationale": rationale,
    })
    return {
        "model": "phi3.5:3.8b-mini-instruct-q4_K_M",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "total_duration": 500_000_000,
        "load_duration": 50_000_000,
        "prompt_eval_count": 120,
        "eval_count": 60,
    }


def _make_client(model="phi3.5:test", **kwargs) -> LLMClassifier:
    kwargs.setdefault("rate_limit_per_minute", 60)
    kwargs.setdefault("cache_ttl_seconds", 3600)
    kwargs.setdefault("cache_max_entries", 100)
    return LLMClassifier(model=model, endpoint="http://test.ollama:11434", **kwargs)


class TestStartup:
    @pytest.mark.asyncio
    @respx.mock
    async def test_startup_ok_when_model_pulled(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}, {"name": "other:x"}]},
        )
        client = _make_client()
        await client.startup()
        assert client.loaded

    @pytest.mark.asyncio
    @respx.mock
    async def test_startup_fails_when_model_missing(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "other:x"}]},
        )
        client = _make_client()
        with pytest.raises(LLMUnavailableError, match="not pulled"):
            await client.startup()
        assert not client.loaded

    @pytest.mark.asyncio
    @respx.mock
    async def test_startup_fails_when_ollama_unreachable(self):
        respx.get("http://test.ollama:11434/api/tags").mock(
            side_effect=httpx.ConnectError("refused"),
        )
        client = _make_client()
        with pytest.raises(LLMUnavailableError, match="Cannot reach"):
            await client.startup()

    @pytest.mark.asyncio
    @respx.mock
    async def test_startup_matches_unqualified_latest(self):
        """Advisor A.4 Bug #1: Ollama canonicalizes `foo` → `foo:latest` in
        /api/tags output. Config `phi3.5:test` should match available
        `phi3.5:test:latest` and vice versa."""
        # Scenario A: config has ":latest", Ollama lists bare name
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        client = _make_client(model="phi3.5:test:latest")
        await client.startup()
        assert client.loaded

    @pytest.mark.asyncio
    @respx.mock
    async def test_startup_matches_bare_against_latest(self):
        # Scenario B: config has bare name, Ollama lists ":latest"
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test:latest"}]},
        )
        client = _make_client(model="phi3.5:test")
        await client.startup()
        assert client.loaded


class TestClassify:
    @pytest.mark.asyncio
    @respx.mock
    async def test_happy_path_returns_llm_result(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        respx.post("http://test.ollama:11434/api/chat").respond(
            200, json=_ollama_chat_response(
                domain="reasoning", complexity="complex", privacy="confidential",
                rationale="Contains sensitive medical context",
            ),
        )
        client = _make_client()
        await client.startup()
        result = await client.classify("I need help with my diagnosis")
        assert result is not None
        assert result.domain == "reasoning"
        assert result.complexity == "complex"
        assert result.privacy == "confidential"
        assert result.rationale == "Contains sensitive medical context"
        assert result.confidence["privacy"] == 0.95

    @pytest.mark.asyncio
    @respx.mock
    async def test_cache_hit_skips_http(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        chat_route = respx.post("http://test.ollama:11434/api/chat").respond(
            200, json=_ollama_chat_response(privacy="public"),
        )
        client = _make_client()
        await client.startup()

        first = await client.classify("what time is it")
        second = await client.classify("what time is it")
        assert first is not None and second is not None
        assert first == second  # same LLMResult (pydantic __eq__ on fields)
        assert chat_route.call_count == 1  # only the first hit went to the wire

    @pytest.mark.asyncio
    async def test_classify_returns_none_before_startup(self):
        client = _make_client()
        result = await client.classify("anything")
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_invalid_json_returns_none(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        respx.post("http://test.ollama:11434/api/chat").respond(
            200, json={
                "message": {"role": "assistant", "content": "not valid json"},
            },
        )
        client = _make_client()
        await client.startup()
        assert await client.classify("x") is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_missing_field_returns_none(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        # JSON parses fine but missing "privacy" field
        content = json.dumps({"domain": "chat", "complexity": "simple"})
        respx.post("http://test.ollama:11434/api/chat").respond(
            200, json={"message": {"role": "assistant", "content": content}},
        )
        client = _make_client()
        await client.startup()
        assert await client.classify("x") is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_unknown_label_returns_none(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        # "confidential_ish" is not a valid Privacy Literal value
        content = json.dumps({
            "domain": "chat", "complexity": "simple", "privacy": "confidential_ish",
        })
        respx.post("http://test.ollama:11434/api/chat").respond(
            200, json={"message": {"role": "assistant", "content": content}},
        )
        client = _make_client()
        await client.startup()
        assert await client.classify("x") is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error_returns_none(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        respx.post("http://test.ollama:11434/api/chat").mock(
            side_effect=httpx.ConnectError("down"),
        )
        client = _make_client()
        await client.startup()
        assert await client.classify("x") is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_5xx_returns_none(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        respx.post("http://test.ollama:11434/api/chat").respond(500, text="server error")
        client = _make_client()
        await client.startup()
        assert await client.classify("x") is None


class TestRateLimit:
    @pytest.mark.asyncio
    @respx.mock
    async def test_rate_limit_blocks_after_max(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        respx.post("http://test.ollama:11434/api/chat").respond(
            200, json=_ollama_chat_response(),
        )
        # Allow only 2 per minute
        client = _make_client(rate_limit_per_minute=2)
        await client.startup()

        # Cache misses force actual HTTP calls; use distinct texts
        a = await client.classify("prompt one")
        b = await client.classify("prompt two")
        c = await client.classify("prompt three")  # rate-limited
        assert a is not None and b is not None
        assert c is None

    @pytest.mark.asyncio
    async def test_limiter_isolated_unit(self):
        limiter = _SlidingWindowLimiter(max_per_minute=3)
        assert await limiter.try_acquire()
        assert await limiter.try_acquire()
        assert await limiter.try_acquire()
        assert not await limiter.try_acquire()  # 4th blocks

    @pytest.mark.asyncio
    async def test_limiter_zero_always_refuses(self):
        limiter = _SlidingWindowLimiter(max_per_minute=0)
        assert not await limiter.try_acquire()


class TestCacheKeyIncludesModel:
    """Advisor A.4 Bug #3: cache keys must include model name — swapping
    models must not surface stale entries from the prior model."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_different_models_cache_independently(self):
        # Set up tags for both models
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}, {"name": "llama:test"}]},
        )
        # Each model's chat endpoint returns a distinct canned classification
        def chat_route_phi(request):
            return httpx.Response(200, json=_ollama_chat_response(
                domain="chat", privacy="public", rationale="phi-response",
            ))

        def chat_route_llama(request):
            return httpx.Response(200, json=_ollama_chat_response(
                domain="reasoning", privacy="confidential", rationale="llama-response",
            ))

        # First client with phi
        respx.post("http://test.ollama:11434/api/chat").mock(side_effect=chat_route_phi)
        client_phi = _make_client(model="phi3.5:test")
        await client_phi.startup()
        result_phi = await client_phi.classify("the same prompt")
        assert result_phi is not None and result_phi.rationale == "phi-response"

        # Switch to llama with a NEW client instance (fresh cache) — but
        # the real worry is: if someone reused one cache across clients,
        # would the second lookup find phi's cached result? Our per-client
        # cache + model-keyed hash guarantees no.
        respx.post("http://test.ollama:11434/api/chat").mock(side_effect=chat_route_llama)
        client_llama = _make_client(model="llama:test")
        await client_llama.startup()
        result_llama = await client_llama.classify("the same prompt")
        assert result_llama is not None and result_llama.rationale == "llama-response"
        # Different models, different cache entries — no stale hit.

    def test_cache_key_hash_differs_by_model(self):
        a = LLMClassifier(model="phi3.5:test", endpoint="http://x")
        b = LLMClassifier(model="llama:test", endpoint="http://x")
        assert a._cache_key("hello") != b._cache_key("hello")


class TestCacheTTL:
    @pytest.mark.asyncio
    @respx.mock
    async def test_cache_size_grows_with_distinct_prompts(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        respx.post("http://test.ollama:11434/api/chat").respond(
            200, json=_ollama_chat_response(),
        )
        client = _make_client(cache_max_entries=100)
        await client.startup()
        await client.classify("one")
        await client.classify("two")
        await client.classify("three")
        assert client.cache_size == 3

    @pytest.mark.asyncio
    @respx.mock
    async def test_cache_eviction_lru(self):
        respx.get("http://test.ollama:11434/api/tags").respond(
            200, json={"models": [{"name": "phi3.5:test"}]},
        )
        respx.post("http://test.ollama:11434/api/chat").respond(
            200, json=_ollama_chat_response(),
        )
        client = _make_client(cache_max_entries=2)
        await client.startup()
        await client.classify("a")
        await client.classify("b")
        await client.classify("c")  # evicts "a"
        assert client.cache_size == 2
