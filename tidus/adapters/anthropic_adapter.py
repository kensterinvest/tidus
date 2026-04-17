"""Anthropic adapter — Claude Opus, Sonnet, and Haiku families.

Environment:
    ANTHROPIC_API_KEY: Anthropic API key (required)
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
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("anthropic not installed: pip install anthropic") from exc
        settings = get_settings()
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


@register_adapter
class AnthropicAdapter(AbstractModelAdapter):
    vendor = "anthropic"
    supported_model_ids = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]

    async def complete(self, model_id: str, task) -> AdapterResponse:
        client = _get_client()
        settings = get_settings()

        # Separate system message if present
        system = None
        messages = []
        for msg in task.messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                messages.append(msg)

        kwargs: dict = {
            "model": model_id,
            "max_tokens": task.estimated_output_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        async def do_call():
            try:
                return await client.messages.create(**kwargs)
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

        # Aggregate all text blocks — Anthropic returns a list for multi-block
        # responses (tool use + text). Previously we dropped everything except
        # the first block, losing content. See Fix 18 / review report.
        text_blocks = [
            block.text for block in (response.content or [])
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]
        content = "\n".join(text_blocks)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        log.info(
            "anthropic_complete",
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
            finish_reason=response.stop_reason or "stop",
            raw=response.model_dump(),
        )

    async def stream_complete(self, model_id: str, task) -> AsyncIterator[str]:
        client = _get_client()

        system = None
        messages = []
        for msg in task.messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                messages.append(msg)

        kwargs: dict = {
            "model": model_id,
            "max_tokens": task.estimated_output_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def health_check(self, model_id: str) -> bool:
        try:
            client = _get_client()
            response = await client.messages.create(
                model=model_id,
                max_tokens=5,
                messages=[{"role": "user", "content": _PROBE_PROMPT}],
            )
            return bool(response.content)
        except Exception:
            return False

    async def count_tokens(self, model_id: str, messages: list[dict]) -> int:
        try:
            client = _get_client()
            # Filter out system messages for count_tokens
            msgs = [m for m in messages if m.get("role") != "system"]
            if not msgs:
                msgs = [{"role": "user", "content": ""}]
            response = await client.messages.count_tokens(model=model_id, messages=msgs)
            return response.input_tokens
        except Exception:
            # Fallback: approximate 4 chars per token
            total = sum(len(m.get("content", "")) for m in messages)
            return max(1, total // 4)
