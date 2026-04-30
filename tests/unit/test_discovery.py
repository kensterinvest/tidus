"""Unit tests for vendor model discovery.

Covers the OpenAI-compatible source filtering / canonicalization, the
Anthropic source's date-suffix stripping, and the runner's first-seen
diff against a registry + persistence to a JSON sidecar.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import MockTransport, Request, Response

from tidus.sync.discovery.anthropic import AnthropicDiscoverySource
from tidus.sync.discovery.base import DiscoveredModel
from tidus.sync.discovery.google import GoogleDiscoverySource
from tidus.sync.discovery.openai_compatible import (
    deepseek_source,
    mistral_source,
    openai_source,
    xai_source,
)
from tidus.sync.discovery.runner import DiscoveryRunner

# ── OpenAI-compatible source: filtering + canonicalization ────────────────────

def _patch_httpx(monkeypatch, route_handler):
    """Replace httpx.AsyncClient with one whose underlying transport is a
    MockTransport — letting us assert the request and synthesize responses
    without hitting real vendor APIs."""
    import httpx

    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = MockTransport(route_handler)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


@pytest.mark.asyncio
async def test_openai_compatible_filters_unwanted_variants(monkeypatch):
    captured: dict = {}

    def handler(request: Request) -> Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        return Response(200, json={
            "data": [
                {"id": "gpt-4.1-mini", "owned_by": "openai"},
                {"id": "text-embedding-3-large", "owned_by": "openai"},  # filtered
                {"id": "whisper-1", "owned_by": "openai"},                # filtered
                {"id": "tts-1", "owned_by": "openai"},                    # filtered
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "ft:gpt-4o:org::abc", "owned_by": "openai"},       # filtered (fine-tuned)
            ]
        })

    _patch_httpx(monkeypatch, handler)
    src = openai_source(api_key="sk-test")
    result = await src.list_models()

    assert {m.model_id for m in result} == {"gpt-4.1-mini", "gpt-4o"}
    assert captured["url"] == "https://api.openai.com/v1/models"
    assert captured["auth"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_mistral_strips_aliases(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, json={
            "data": [
                {"id": "mistral-large-latest"},
                {"id": "mistral-medium-2407"},
                {"id": "codestral-latest"},
            ]
        })

    _patch_httpx(monkeypatch, handler)
    src = mistral_source(api_key="m-key")
    result = await src.list_models()
    assert {m.model_id for m in result} == {
        "mistral-large", "mistral-medium", "codestral",
    }


@pytest.mark.asyncio
async def test_deepseek_canonicalizes_aliases(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, json={
            "data": [
                {"id": "deepseek-chat"},
                {"id": "deepseek-reasoner"},
                {"id": "deepseek-experimental"},  # passthrough
            ]
        })

    _patch_httpx(monkeypatch, handler)
    src = deepseek_source(api_key="ds-key")
    result = await src.list_models()
    by_canonical = {m.model_id: m.vendor_id for m in result}
    assert by_canonical == {
        "deepseek-v3": "deepseek-chat",
        "deepseek-r1": "deepseek-reasoner",
        "deepseek-experimental": "deepseek-experimental",
    }


@pytest.mark.asyncio
async def test_xai_strips_date_suffix(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, json={"data": [{"id": "grok-4-0709"}, {"id": "grok-3"}]})

    _patch_httpx(monkeypatch, handler)
    src = xai_source(api_key="x-key")
    result = await src.list_models()
    assert {m.model_id for m in result} == {"grok-4", "grok-3"}


@pytest.mark.asyncio
async def test_source_returns_empty_on_http_error(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(500, json={"error": "boom"})

    _patch_httpx(monkeypatch, handler)
    src = openai_source(api_key="sk-test")
    result = await src.list_models()
    assert result == []


@pytest.mark.asyncio
async def test_source_returns_empty_on_parse_error(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(200, content=b"not-json")

    _patch_httpx(monkeypatch, handler)
    src = openai_source(api_key="sk-test")
    result = await src.list_models()
    assert result == []


def test_source_unavailable_when_no_api_key():
    src = openai_source(api_key="")
    assert src.is_available is False


# ── Google source: REST shape, generateContent gate, multi-page handling ─────

@pytest.mark.asyncio
async def test_google_filters_to_generate_content_models(monkeypatch):
    captured: dict = {}

    def handler(request: Request) -> Response:
        captured["url"] = str(request.url)
        captured["key_param"] = request.url.params.get("key")
        return Response(200, json={
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                    "inputTokenLimit": 2000000,
                    "outputTokenLimit": 8192,
                },
                {
                    # countTokens-only — must be filtered out
                    "name": "models/text-tokens-only",
                    "displayName": "Tokens",
                    "supportedGenerationMethods": ["countTokens"],
                },
                {
                    # Embedding model — must be filtered even if it lists generateContent
                    "name": "models/text-embedding-004",
                    "displayName": "Text Embedding 004",
                    "supportedGenerationMethods": ["generateContent", "embedContent"],
                },
                {
                    "name": "models/gemini-2.5-flash-latest",
                    "displayName": "Gemini 2.5 Flash (latest)",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    # AQA — Q&A specific, filter
                    "name": "models/aqa",
                    "displayName": "Attributed QA",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]
        })

    _patch_httpx(monkeypatch, handler)
    src = GoogleDiscoverySource(api_key="g-key")
    result = await src.list_models()

    by_canonical = {m.model_id: m.vendor_id for m in result}
    # Only the two real generation-capable Gemini entries pass.
    # 'gemini-2.5-flash-latest' must canonicalize to 'gemini-2.5-flash'.
    assert by_canonical == {
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash-latest",
    }
    # `?key=` query-param auth (NOT a Bearer header)
    assert captured["key_param"] == "g-key"
    assert "/v1beta/models" in captured["url"]


@pytest.mark.asyncio
async def test_google_paginates_until_no_token(monkeypatch):
    """Multi-page Google response — runner must follow nextPageToken until exhausted."""
    pages = [
        {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ],
            "nextPageToken": "page-2",
        },
        {
            "models": [
                {
                    "name": "models/gemini-3.1-pro",
                    "displayName": "Gemini 3.1 Pro",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ],
        },
    ]
    page_idx = {"i": 0}

    def handler(request: Request) -> Response:
        body = pages[page_idx["i"]]
        page_idx["i"] += 1
        return Response(200, json=body)

    _patch_httpx(monkeypatch, handler)
    src = GoogleDiscoverySource(api_key="g-key")
    result = await src.list_models()
    assert {m.model_id for m in result} == {"gemini-2.5-pro", "gemini-3.1-pro"}
    assert page_idx["i"] == 2  # second call followed nextPageToken


@pytest.mark.asyncio
async def test_google_returns_empty_on_invalid_key(monkeypatch):
    def handler(request: Request) -> Response:
        return Response(400, json={
            "error": {"code": 400, "message": "API key not valid"}
        })

    _patch_httpx(monkeypatch, handler)
    src = GoogleDiscoverySource(api_key="g-bad")
    assert await src.list_models() == []


def test_google_unavailable_when_no_api_key():
    assert GoogleDiscoverySource(api_key="").is_available is False


# ── Anthropic source: date-suffix stripping + auth header shape ───────────────

@pytest.mark.asyncio
async def test_anthropic_strips_date_suffix(monkeypatch):
    captured: dict = {}

    def handler(request: Request) -> Response:
        captured["url"] = str(request.url)
        captured["x_api_key"] = request.headers.get("x-api-key")
        captured["version"] = request.headers.get("anthropic-version")
        return Response(200, json={
            "data": [
                {
                    "id": "claude-opus-4-7-20260420",
                    "type": "model",
                    "display_name": "Claude Opus 4.7",
                    "created_at": "2026-04-20T00:00:00Z",
                },
                {
                    "id": "claude-haiku-4-5-20260101",
                    "type": "model",
                    "display_name": "Claude Haiku 4.5",
                    "created_at": "2026-01-01T00:00:00Z",
                },
            ]
        })

    _patch_httpx(monkeypatch, handler)
    src = AnthropicDiscoverySource(api_key="a-key")
    result = await src.list_models()
    by_canonical = {m.model_id: m.vendor_id for m in result}
    assert by_canonical == {
        "claude-opus-4-7": "claude-opus-4-7-20260420",
        "claude-haiku-4-5": "claude-haiku-4-5-20260101",
    }
    assert captured["x_api_key"] == "a-key"
    assert captured["version"] == "2023-06-01"


# ── Runner: registry diff + first-seen state ──────────────────────────────────

class StubSource:
    """In-memory DiscoverySource — feeds the runner deterministic data."""

    def __init__(self, name: str, vendor: str, models: list[DiscoveredModel], available: bool = True) -> None:
        self._name, self._vendor, self._models, self._available = name, vendor, models, available

    @property
    def source_name(self) -> str: return self._name
    @property
    def vendor(self) -> str: return self._vendor
    @property
    def is_available(self) -> bool: return self._available

    async def list_models(self) -> list[DiscoveredModel]:
        return list(self._models)


def _mk(model_id: str, vendor: str, vendor_id: str | None = None) -> DiscoveredModel:
    return DiscoveredModel(
        model_id=model_id,
        vendor_id=vendor_id or model_id,
        vendor=vendor,
        display_name=f"{vendor} {model_id}",
        source_name=f"{vendor}-models",
        retrieved_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_runner_first_seen_separates_new_vs_known(tmp_path: Path):
    state = tmp_path / "discovered.json"

    # First run — both models are new to us, neither is in the registry.
    src = StubSource("openai-models", "openai", [_mk("gpt-5", "openai"), _mk("gpt-4.1", "openai")])
    runner = DiscoveryRunner([src], state_path=state, registry_model_ids={"gpt-4.1"})
    report1 = await runner.run()

    # gpt-4.1 is in registry → not flagged. gpt-5 is genuinely new → new_this_run.
    assert {m.model_id for m in report1.new_this_run} == {"gpt-5"}
    assert report1.pending_review == []
    assert report1.removed_from_vendor == []
    assert state.exists()

    # Second run with same data — gpt-5 is still pending review (not in registry,
    # but no longer "new this run" because we've seen it before).
    runner2 = DiscoveryRunner([src], state_path=state, registry_model_ids={"gpt-4.1"})
    report2 = await runner2.run()

    assert {m.model_id for m in report2.new_this_run} == set()
    assert {m.model_id for m in report2.pending_review} == {"gpt-5"}
    assert report2.removed_from_vendor == []


@pytest.mark.asyncio
async def test_runner_records_models_absent_this_cycle(tmp_path: Path):
    state = tmp_path / "discovered.json"

    src1 = StubSource("openai-models", "openai", [_mk("gpt-5", "openai"), _mk("gpt-old", "openai")])
    runner1 = DiscoveryRunner([src1], state_path=state, registry_model_ids=set())
    await runner1.run()

    # On the second run, vendor stops returning gpt-old.
    src2 = StubSource("openai-models", "openai", [_mk("gpt-5", "openai")])
    runner2 = DiscoveryRunner([src2], state_path=state, registry_model_ids=set())
    report = await runner2.run()

    assert "gpt-old" in report.removed_from_vendor
    assert {m.model_id for m in report.pending_review} == {"gpt-5"}


@pytest.mark.asyncio
async def test_runner_skips_unavailable_sources(tmp_path: Path):
    src_ok = StubSource("openai-models", "openai", [_mk("gpt-5", "openai")])
    src_skipped = StubSource("anthropic-models", "anthropic", [], available=False)
    runner = DiscoveryRunner([src_ok, src_skipped], state_path=tmp_path / "s.json", registry_model_ids=set())
    report = await runner.run()

    assert report.sources_run == ["openai-models"]
    assert report.sources_skipped == ["anthropic-models"]


@pytest.mark.asyncio
async def test_runner_state_persists_across_runs(tmp_path: Path):
    state = tmp_path / "discovered.json"
    src = StubSource("openai-models", "openai", [_mk("gpt-5", "openai")])
    runner = DiscoveryRunner([src], state_path=state, registry_model_ids=set())
    await runner.run()

    written = json.loads(state.read_text())
    assert "gpt-5" in written
    entry = written["gpt-5"]
    assert entry["vendor"] == "openai"
    assert entry["in_registry"] is False
    assert entry["first_seen"] == entry["last_seen"]


@pytest.mark.asyncio
async def test_runner_survives_source_exceptions(tmp_path: Path):
    """A single source raising must not poison the rest of the run."""

    class BoomSource(StubSource):
        async def list_models(self) -> list[DiscoveredModel]:
            raise RuntimeError("kaboom")

    boom = BoomSource("xai-models", "xai", [])
    ok = StubSource("openai-models", "openai", [_mk("gpt-5", "openai")])
    runner = DiscoveryRunner([boom, ok], state_path=tmp_path / "s.json", registry_model_ids=set())
    report = await runner.run()

    assert {m.model_id for m in report.new_this_run} == {"gpt-5"}


@pytest.mark.asyncio
async def test_runner_corrupt_state_is_treated_as_empty(tmp_path: Path):
    state = tmp_path / "discovered.json"
    state.write_text("{not valid json")
    src = StubSource("openai-models", "openai", [_mk("gpt-5", "openai")])
    runner = DiscoveryRunner([src], state_path=state, registry_model_ids=set())
    report = await runner.run()

    # Corrupt prior state → we treat it as a fresh start, so gpt-5 IS new.
    assert {m.model_id for m in report.new_this_run} == {"gpt-5"}


# ── Report integration: discovery section appears when populated ──────────────

def test_pricing_report_renders_discovery_section_when_populated():
    """The pricing report should include the discovery section iff the
    DiscoveryReport has findings; empty reports must not pollute output.
    """
    from datetime import date as date_t

    from tidus.reporting.pricing_report import PricingReport, PricingReportGenerator
    from tidus.sync.discovery.runner import DiscoveryReport

    populated = DiscoveryReport(
        generated_at=datetime.now(UTC),
        sources_run=["openai-models", "anthropic-models"],
        sources_skipped=[],
        new_this_run=[_mk("gpt-5", "openai", "gpt-5-2026-04-30")],
        pending_review=[_mk("claude-haiku-4-6", "anthropic")],
        removed_from_vendor=["legacy-model"],
        total_discovered=3,
    )
    report = PricingReport(
        generated_at=datetime.now(UTC),
        report_date=date_t(2026, 4, 30),
        current_revision_id="rev-x",
        base_revision_id=None,
        new_models=[],
        price_changes=[],
        stale_models=[],
        total_models=43,
        discovery_report=populated,
    )

    md = PricingReportGenerator(session_factory=None)._render_markdown(report, {})
    assert "🔎 Vendor-Discovered Models" in md
    assert "First-seen this run" in md
    assert "gpt-5" in md and "gpt-5-2026-04-30" in md
    assert "Backlog" in md
    assert "claude-haiku-4-6" in md
    assert "Absent this run" in md
    assert "legacy-model" in md


def test_pricing_report_omits_discovery_section_when_empty():
    from datetime import date as date_t

    from tidus.reporting.pricing_report import PricingReport, PricingReportGenerator
    from tidus.sync.discovery.runner import DiscoveryReport

    empty = DiscoveryReport(
        generated_at=datetime.now(UTC),
        sources_run=["openai-models"],
        sources_skipped=[],
        new_this_run=[],
        pending_review=[],
        removed_from_vendor=[],
        total_discovered=0,
    )
    report = PricingReport(
        generated_at=datetime.now(UTC),
        report_date=date_t(2026, 4, 30),
        current_revision_id="rev-x",
        base_revision_id=None,
        new_models=[],
        price_changes=[],
        stale_models=[],
        total_models=43,
        discovery_report=empty,
    )

    md = PricingReportGenerator(session_factory=None)._render_markdown(report, {})
    assert "Vendor-Discovered Models" not in md
