"""Google adapter — Gemini 3.1 Pro, 3.1 Flash, and Nano (local).

Environment:
    GOOGLE_API_KEY: Google AI Studio API key (required for cloud models)
    Gemini Nano is device-local — no key required.
"""

from __future__ import annotations

import time
from typing import AsyncIterator

import structlog

from tidus.adapters.base import AbstractModelAdapter, AdapterResponse, register_adapter
from tidus.settings import get_settings

log = structlog.get_logger(__name__)

_PROBE_PROMPT = "hi"
_genai = None


def _get_genai():
    global _genai
    if _genai is None:
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai not installed: pip install google-generativeai"
            ) from exc
        settings = get_settings()
        genai.configure(api_key=settings.google_api_key)
        _genai = genai
    return _genai


def _to_genai_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split system prompt out; convert remaining messages to Gemini format."""
    system = None
    history = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system = content
        elif role == "assistant":
            history.append({"role": "model", "parts": [content]})
        else:
            history.append({"role": "user", "parts": [content]})
    return system, history


@register_adapter
class GoogleAdapter(AbstractModelAdapter):
    vendor = "google"
    supported_model_ids = ["gemini-3.1-pro", "gemini-3.1-flash", "gemini-nano"]

    async def complete(self, model_id: str, task) -> AdapterResponse:
        genai = _get_genai()
        system, history = _to_genai_messages(task.messages)

        kwargs: dict = {}
        if system:
            kwargs["system_instruction"] = system

        model = genai.GenerativeModel(model_id, **kwargs)
        chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])
        last_user = history[-1]["parts"][0] if history else ""

        start = time.monotonic()
        response = await chat.send_message_async(
            last_user,
            generation_config=genai.GenerationConfig(
                max_output_tokens=task.estimated_output_tokens
            ),
        )
        latency_ms = (time.monotonic() - start) * 1000

        content = response.text or ""
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0

        log.info(
            "google_complete",
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 1),
        )
        return AdapterResponse(
            model_id=model_id,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason="stop",
            raw={},
        )

    async def stream_complete(self, model_id: str, task) -> AsyncIterator[str]:
        genai = _get_genai()
        system, history = _to_genai_messages(task.messages)
        kwargs: dict = {}
        if system:
            kwargs["system_instruction"] = system
        model = genai.GenerativeModel(model_id, **kwargs)
        last_user = history[-1]["parts"][0] if history else ""

        response = await model.generate_content_async(last_user, stream=True)
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def health_check(self, model_id: str) -> bool:
        try:
            genai = _get_genai()
            model = genai.GenerativeModel(model_id)
            response = await model.generate_content_async(_PROBE_PROMPT)
            return bool(response.text)
        except Exception:
            return False

    async def count_tokens(self, model_id: str, messages: list[dict]) -> int:
        try:
            genai = _get_genai()
            model = genai.GenerativeModel(model_id)
            _, history = _to_genai_messages(messages)
            result = model.count_tokens(history)
            return result.total_tokens
        except Exception:
            total = sum(len(m.get("content", "")) for m in messages)
            return max(1, total // 4)
