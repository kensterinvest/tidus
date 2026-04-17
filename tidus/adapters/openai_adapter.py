"""OpenAI adapter — GPT-4.1, o3, o4-mini, GPT-4o mini, GPT-OSS 120B, Codex family.

Environment:
    OPENAI_API_KEY: OpenAI API key (required)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import structlog

from tidus.adapters.base import (
    AbstractModelAdapter,
    AdapterError,
    AdapterResponse,
    register_adapter,
    translate_vendor_exception,
    with_retry,
)
from tidus.settings import get_settings

log = structlog.get_logger(__name__)

_PROBE_PROMPT = "hi"
_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("openai not installed: pip install openai") from exc
        settings = get_settings()
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


@register_adapter
class OpenAIAdapter(AbstractModelAdapter):
    vendor = "openai"
    supported_model_ids = [
        "o3", "o4-mini", "gpt-4.1", "gpt-4o-mini", "gpt-oss-120b",
        "gpt-5-codex", "codex-mini-latest",
    ]

    async def complete(self, model_id: str, task) -> AdapterResponse:
        client = _get_client()
        settings = get_settings()

        async def do_call():
            try:
                return await client.chat.completions.create(
                    model=model_id,
                    messages=task.messages,
                    max_completion_tokens=task.estimated_output_tokens,
                )
            except AdapterError:
                raise
            except Exception as exc:
                raise translate_vendor_exception(exc) from exc

        start = time.monotonic()
        response = await with_retry(
            do_call,
            timeout_seconds=settings.adapter_timeout_seconds,
            max_attempts=settings.adapter_max_retries,
            base_delay_seconds=settings.adapter_base_delay_seconds,
        )
        latency_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        content = choice.message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        finish_reason = choice.finish_reason or "stop"

        log.info(
            "openai_complete",
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
            finish_reason=finish_reason,
            raw=response.model_dump(),
        )

    async def stream_complete(self, model_id: str, task) -> AsyncIterator[str]:
        client = _get_client()
        stream = await client.chat.completions.create(
            model=model_id,
            messages=task.messages,
            max_completion_tokens=task.estimated_output_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def health_check(self, model_id: str) -> bool:
        try:
            client = _get_client()
            response = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": _PROBE_PROMPT}],
                max_completion_tokens=5,
            )
            return bool(response.choices)
        except Exception:
            return False

    async def count_tokens(self, model_id: str, messages: list[dict]) -> int:
        try:
            import tiktoken  # type: ignore[import-untyped]
            # o200k_base for GPT-4o and newer models, cl100k for older
            enc_name = "o200k_base" if any(
                m in model_id for m in ("gpt-4o", "gpt-4.1", "o3", "o4", "codex", "gpt-5")
            ) else "cl100k_base"
            enc = tiktoken.get_encoding(enc_name)
            total = 0
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
                total += len(enc.encode(content)) + 3
            return total
        except Exception:
            total = sum(len(m.get("content", "")) for m in messages)
            return max(1, total // 4)
