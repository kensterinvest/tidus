"""Unit tests for the OpenRouter universal execution adapter + resolver."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tidus.adapters import openrouter_adapter
from tidus.adapters.adapter_factory import get_adapter, resolve_adapter
from tidus.adapters.openrouter_adapter import OpenRouterAdapter


def _spec(model_id="gpt-4o", vendor="openai", route_id=None):
    return SimpleNamespace(model_id=model_id, vendor=vendor, route_id=route_id)


# ── resolver ─────────────────────────────────────────────────────────────────


class TestResolveAdapter:
    def test_route_id_uses_openrouter_adapter_with_route_id(self):
        spec = _spec(
            model_id="nemotron-3-ultra-550b-a55b", vendor="nvidia",
            route_id="nvidia/nemotron-3-ultra-550b-a55b",
        )
        adapter, exec_id = resolve_adapter(spec)
        assert isinstance(adapter, OpenRouterAdapter)
        assert exec_id == "nvidia/nemotron-3-ultra-550b-a55b"

    def test_native_vendor_uses_native_adapter_with_model_id(self):
        spec = _spec(model_id="gpt-4o", vendor="openai", route_id=None)
        adapter, exec_id = resolve_adapter(spec)
        assert adapter is get_adapter("openai")
        assert exec_id == "gpt-4o"

    def test_unknown_vendor_no_route_id_raises(self):
        # No native adapter AND no route_id → cannot serve → KeyError (501 path).
        spec = _spec(model_id="mystery", vendor="acme", route_id=None)
        with pytest.raises(KeyError):
            resolve_adapter(spec)


# ── adapter ──────────────────────────────────────────────────────────────────


class TestOpenRouterAdapter:
    def test_registered_under_openrouter_vendor(self):
        assert isinstance(get_adapter("openrouter"), OpenRouterAdapter)
        assert OpenRouterAdapter.vendor == "openrouter"

    @pytest.mark.asyncio
    async def test_count_tokens_returns_positive_int(self):
        n = await OpenRouterAdapter().count_tokens(
            "nvidia/nemotron-3-ultra-550b-a55b",
            [{"role": "user", "content": "hello world"}],
        )
        assert isinstance(n, int) and n > 0

    @pytest.mark.asyncio
    async def test_complete_calls_openrouter_with_route_id(self, monkeypatch):
        captured = {}

        class _FakeCompletions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="hi from nemotron"),
                        finish_reason="stop",
                    )],
                    usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
                    model_dump=lambda: {"ok": True},
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
        monkeypatch.setattr(openrouter_adapter, "_get_client", lambda: fake_client)

        task = SimpleNamespace(
            messages=[{"role": "user", "content": "hello"}],
            estimated_output_tokens=64,
        )
        resp = await OpenRouterAdapter().complete("nvidia/nemotron-3-ultra-550b-a55b", task)

        # The OpenRouter route id is passed as the model param.
        assert captured["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
        assert resp.content == "hi from nemotron"
        assert resp.input_tokens == 11
        assert resp.output_tokens == 7
        assert resp.model_id == "nvidia/nemotron-3-ultra-550b-a55b"
