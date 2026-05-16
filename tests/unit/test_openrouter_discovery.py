"""Unit tests for OpenRouterDiscoverySource + factory registration."""

from __future__ import annotations

import pytest
from httpx import MockTransport, Request, Response

from tidus.settings import Settings
from tidus.sync.discovery.factory import build_discovery_sources
from tidus.sync.discovery.openrouter import (
    OpenRouterDiscoverySource,
    _vendor_from_or_id,
)


def _patch_httpx_with(monkeypatch, handler):
    import httpx

    real_init = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = MockTransport(handler)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


# ── Vendor extraction ─────────────────────────────────────────────────────────

class TestVendorFromOrId:
    def test_known_prefixes_normalized(self):
        assert _vendor_from_or_id("google/gemini-2.5-pro") == "google"
        assert _vendor_from_or_id("mistralai/mistral-large") == "mistral"
        assert _vendor_from_or_id("x-ai/grok-4") == "xai"

    def test_unknown_prefix_passthrough(self):
        assert _vendor_from_or_id("brand-new-vendor/foo") == "brand-new-vendor"

    def test_no_slash_returns_empty(self):
        assert _vendor_from_or_id("just-a-name") == ""


# ── HTTP fetch ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_models_parses_data_and_captures_pricing_metadata(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, json={
            "data": [
                {
                    "id": "google/gemini-3.5-pro",
                    "name": "Google: Gemini 3.5 Pro",
                    "context_length": 2000000,
                    "architecture": {"input_modalities": ["text", "image"]},
                    "pricing": {"prompt": "0.000002", "completion": "0.00001"},
                },
                {
                    "id": "anthropic/claude-opus-5",
                    "name": "Anthropic: Claude Opus 5",
                    "context_length": 500000,
                    "pricing": {"prompt": "0.00001", "completion": "0.00005"},
                },
            ]
        })

    _patch_httpx_with(monkeypatch, handler)
    models = await OpenRouterDiscoverySource().list_models()

    assert len(models) == 2
    by_id = {m.model_id: m for m in models}

    g = by_id["gemini-3.5-pro"]
    assert g.vendor == "google"
    assert g.vendor_id == "google/gemini-3.5-pro"
    assert g.display_name == "Google: Gemini 3.5 Pro"
    assert g.source_name == "openrouter-discovery"
    assert g.raw_metadata["openrouter_id"] == "google/gemini-3.5-pro"
    assert g.raw_metadata["context_length"] == 2000000
    assert g.raw_metadata["input_modalities"] == ["text", "image"]
    assert g.raw_metadata["pricing"]["prompt"] == "0.000002"
    assert g.raw_metadata["pricing"]["completion"] == "0.00001"

    c = by_id["claude-opus-5"]
    assert c.vendor == "anthropic"


@pytest.mark.asyncio
async def test_variant_suffix_collapses_to_one_entry(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, json={
            "data": [
                {"id": "google/gemini-2.5-pro",        "pricing": {"prompt": "0.000001", "completion": "0.000005"}},
                {"id": "google/gemini-2.5-pro:nitro",  "pricing": {"prompt": "0.000002", "completion": "0.00001"}},
                {"id": "google/gemini-2.5-pro:free",   "pricing": {"prompt": "0",         "completion": "0"}},
            ]
        })

    _patch_httpx_with(monkeypatch, handler)
    models = await OpenRouterDiscoverySource().list_models()
    assert [m.model_id for m in models] == ["gemini-2.5-pro"]


@pytest.mark.asyncio
async def test_no_slash_id_dropped(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, json={
            "data": [
                {"id": "no-slash", "pricing": {"prompt": "0.001", "completion": "0.001"}},
                {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
            ]
        })

    _patch_httpx_with(monkeypatch, handler)
    models = await OpenRouterDiscoverySource().list_models()
    assert [m.model_id for m in models] == ["gpt-4o"]


@pytest.mark.asyncio
async def test_http_failure_returns_empty_list(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(503, text="service unavailable")

    _patch_httpx_with(monkeypatch, handler)
    models = await OpenRouterDiscoverySource().list_models()
    assert models == []


@pytest.mark.asyncio
async def test_malformed_json_returns_empty_list(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, text="not-json{{{")

    _patch_httpx_with(monkeypatch, handler)
    models = await OpenRouterDiscoverySource().list_models()
    assert models == []


@pytest.mark.asyncio
async def test_disabled_source_does_not_hit_network():
    # No httpx patch — would raise on any real network call.
    models = await OpenRouterDiscoverySource(enabled=False).list_models()
    assert models == []


# ── Factory registration ──────────────────────────────────────────────────────

def _settings_with(**overrides) -> Settings:
    defaults = {
        "openai_api_key": "",
        "anthropic_api_key": "",
        "google_api_key": "",
        "mistral_api_key": "",
        "deepseek_api_key": "",
        "xai_api_key": "",
        "openrouter_enabled": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_factory_includes_openrouter_when_no_vendor_keys():
    sources = build_discovery_sources(_settings_with())
    names = [s.source_name for s in sources]
    assert names == ["openrouter-discovery"]


def test_factory_appends_openrouter_after_first_party_sources():
    s = _settings_with(google_api_key="fake", openai_api_key="fake")
    sources = build_discovery_sources(s)
    names = [src.source_name for src in sources]
    # OpenRouter must be last so first-party sources win the dedup race.
    assert names[-1] == "openrouter-discovery"
    assert "openrouter-discovery" in names
    assert "google-models" in names


def test_factory_omits_openrouter_when_disabled():
    sources = build_discovery_sources(_settings_with(openrouter_enabled=False))
    assert sources == []
