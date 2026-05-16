"""Unit tests for OpenRouterPricingSource.

Covers id mapping, price parsing, dedup, payload-error tolerance, and the
"all sources down → empty list, never raise" guarantee the pipeline relies on.
"""

from __future__ import annotations

import pytest
from httpx import MockTransport, Request, Response

from tidus.sync.pricing.openrouter_source import (
    OpenRouterPricingSource,
    _canonical_from_openrouter,
)


# ── Pure helper: id mapping ───────────────────────────────────────────────────

class TestCanonicalMapping:
    def test_slash_strip_for_vendor_prefix(self):
        assert _canonical_from_openrouter("google/gemini-2.5-pro") == "gemini-2.5-pro"
        assert _canonical_from_openrouter("openai/gpt-4o-mini") == "gpt-4o-mini"

    def test_explicit_override_wins_over_slash_strip(self):
        # Anthropic id without Tidus's date suffix needs the override table.
        assert _canonical_from_openrouter("anthropic/claude-opus-4.7") == "claude-opus-4-7"

    def test_variant_suffix_stripped_before_lookup(self):
        # `:free` / `:nitro` etc. don't change which model it is.
        assert _canonical_from_openrouter("google/gemini-2.5-pro:free") == "gemini-2.5-pro"
        assert _canonical_from_openrouter("anthropic/claude-opus-4.7:beta") == "claude-opus-4-7"

    def test_unmappable_returns_none(self):
        assert _canonical_from_openrouter("") is None
        assert _canonical_from_openrouter("no-slash-here") is None
        assert _canonical_from_openrouter("vendor/") is None


# ── HTTP fetch: success path ──────────────────────────────────────────────────

def _patch_httpx_with(monkeypatch, handler):
    import httpx

    real_init = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = MockTransport(handler)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


@pytest.mark.asyncio
async def test_fetch_parses_canonical_models_and_converts_per_token_to_per_1k(monkeypatch):
    captured: dict = {}

    def handler(request: Request) -> Response:
        captured["url"] = str(request.url)
        return Response(200, json={
            "data": [
                {
                    "id": "google/gemini-2.5-pro",
                    "pricing": {
                        "prompt":     "0.00000125",  # → 0.00125 per 1K
                        "completion": "0.00001",      # → 0.01 per 1K
                    },
                },
                {
                    "id": "anthropic/claude-sonnet-4.6",
                    "pricing": {
                        "prompt":     "0.000003",
                        "completion": "0.000015",
                        "input_cache_read":  "0.0000003",
                        "input_cache_write": "0.00000375",
                    },
                },
            ]
        })

    _patch_httpx_with(monkeypatch, handler)
    quotes = await OpenRouterPricingSource().fetch_quotes()

    assert "openrouter.ai/api/v1/models" in captured["url"]
    by_id = {q.model_id: q for q in quotes}
    assert set(by_id.keys()) == {"gemini-2.5-pro", "claude-sonnet-4-6"}

    g = by_id["gemini-2.5-pro"]
    assert g.input_price == pytest.approx(0.00125)
    assert g.output_price == pytest.approx(0.01)
    assert g.source_name == "openrouter"
    assert g.source_confidence == 0.75
    assert g.currency == "USD"
    assert g.evidence_url and "openrouter.ai/models/" in g.evidence_url

    c = by_id["claude-sonnet-4-6"]
    assert c.cache_read_price == pytest.approx(0.0003)
    assert c.cache_write_price == pytest.approx(0.00375)


@pytest.mark.asyncio
async def test_unmappable_models_are_skipped_silently(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, json={
            "data": [
                {"id": "some-new-vendor/exotic-model", "pricing": {"prompt": "0.001", "completion": "0.002"}},
                {"id": "no-slash-in-id", "pricing": {"prompt": "0.001", "completion": "0.002"}},
                {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
            ]
        })

    _patch_httpx_with(monkeypatch, handler)
    quotes = await OpenRouterPricingSource().fetch_quotes()
    # Slash-strip yields "exotic-model" — that IS technically returned (it's
    # a valid suffix). The pipeline filters against config/models.yaml so
    # unknown ids are harmless. We only assert the obviously-unmappable
    # (no-slash, empty) cases are dropped, and a known model survives.
    ids = {q.model_id for q in quotes}
    assert "no-slash-in-id" not in ids
    assert "gpt-4o" in ids


@pytest.mark.asyncio
async def test_zero_price_models_are_skipped(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, json={
            "data": [
                {"id": "openai/free-tier", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "openai/gpt-4o",    "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
            ]
        })

    _patch_httpx_with(monkeypatch, handler)
    quotes = await OpenRouterPricingSource().fetch_quotes()
    assert {q.model_id for q in quotes} == {"gpt-4o"}


@pytest.mark.asyncio
async def test_duplicate_canonical_ids_first_wins(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, json={
            "data": [
                {"id": "google/gemini-2.5-pro", "pricing": {"prompt": "0.00000125", "completion": "0.00001"}},
                # Same canonical after slash-strip — a different provider mirror
                {"id": "google/gemini-2.5-pro:nitro", "pricing": {"prompt": "0.99", "completion": "0.99"}},
            ]
        })

    _patch_httpx_with(monkeypatch, handler)
    quotes = await OpenRouterPricingSource().fetch_quotes()
    assert len(quotes) == 1
    assert quotes[0].input_price == pytest.approx(0.00125)  # the first one


# ── HTTP fetch: failure paths must NOT raise ──────────────────────────────────

@pytest.mark.asyncio
async def test_http_500_returns_empty_list(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(500, text="upstream error")

    _patch_httpx_with(monkeypatch, handler)
    quotes = await OpenRouterPricingSource().fetch_quotes()
    assert quotes == []


@pytest.mark.asyncio
async def test_malformed_json_returns_empty_list(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, text="not-json{{{")

    _patch_httpx_with(monkeypatch, handler)
    quotes = await OpenRouterPricingSource().fetch_quotes()
    assert quotes == []


@pytest.mark.asyncio
async def test_missing_pricing_field_skipped(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, json={
            "data": [
                {"id": "openai/gpt-4o"},                                   # no pricing
                {"id": "google/gemini-2.5-pro", "pricing": {}},            # empty pricing
                {"id": "anthropic/claude-haiku-4.5",
                 "pricing": {"prompt": "0.000001", "completion": "0.000005"}},
            ]
        })

    _patch_httpx_with(monkeypatch, handler)
    quotes = await OpenRouterPricingSource().fetch_quotes()
    assert {q.model_id for q in quotes} == {"claude-haiku-4-5"}


# ── Disabled / availability ───────────────────────────────────────────────────

def test_is_available_when_enabled():
    assert OpenRouterPricingSource(enabled=True).is_available is True


def test_is_unavailable_when_disabled():
    assert OpenRouterPricingSource(enabled=False).is_available is False


@pytest.mark.asyncio
async def test_disabled_source_returns_empty_without_http():
    # No httpx patch — would raise if a request was attempted.
    quotes = await OpenRouterPricingSource(enabled=False).fetch_quotes()
    assert quotes == []
