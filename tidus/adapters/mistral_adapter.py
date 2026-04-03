"""Mistral adapter — Mistral Large 3, Medium, Small, Codestral, Devstral.

Environment:
    MISTRAL_API_KEY: Mistral AI API key (required)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import structlog

from tidus.adapters.base import AbstractModelAdapter, AdapterResponse, register_adapter
from tidus.settings import get_settings

log = structlog.get_logger(__name__)

_PROBE_PROMPT = "hi"
_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from mistralai import Mistral  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("mistralai not installed: pip install mistralai") from exc
        settings = get_settings()
        _client = Mistral(api_key=settings.mistral_api_key)
    return _client


@register_adapter
class MistralAdapter(AbstractModelAdapter):
    vendor = "mistral"
    supported_model_ids = [
        "mistral-large-3", "mistral-medium", "mistral-small",
        "codestral", "devstral",
    ]

    async def complete(self, model_id: str, task) -> AdapterResponse:
        client = _get_client()

        start = time.monotonic()
        response = await client.chat.complete_async(
            model=model_id,
            messages=task.messages,
            max_tokens=task.estimated_output_tokens,
        )
        latency_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        content = choice.message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        log.info(
            "mistral_complete",
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
            finish_reason=str(choice.finish_reason or "stop"),
            raw={},
        )

    async def stream_complete(self, model_id: str, task) -> AsyncIterator[str]:
        client = _get_client()
        stream = await client.chat.stream_async(
            model=model_id,
            messages=task.messages,
            max_tokens=task.estimated_output_tokens,
        )
        async for event in stream:
            delta = event.data.choices[0].delta.content
            if delta:
                yield delta

    async def health_check(self, model_id: str) -> bool:
        try:
            client = _get_client()
            response = await client.chat.complete_async(
                model=model_id,
                messages=[{"role": "user", "content": _PROBE_PROMPT}],
                max_tokens=5,
            )
            return bool(response.choices)
        except Exception:
            return False

    async def count_tokens(self, model_id: str, messages: list[dict]) -> int:
        # Mistral doesn't expose a token count endpoint; use sentencepiece approximation
        try:
            from mistral_common.protocol.instruct.messages import (  # type: ignore[import-untyped]
                AssistantMessage,
                SystemMessage,
                UserMessage,
            )
            from mistral_common.protocol.instruct.request import (
                ChatCompletionRequest,  # type: ignore[import-untyped]
            )
            from mistral_common.tokens.tokenizers.mistral import (
                MistralTokenizer,  # type: ignore[import-untyped]
            )

            tokenizer = MistralTokenizer.v3()
            mistral_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
                if role == "system":
                    mistral_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    mistral_messages.append(AssistantMessage(content=content))
                else:
                    mistral_messages.append(UserMessage(content=content))
            request = ChatCompletionRequest(messages=mistral_messages)
            tokens = tokenizer.encode_chat_completion(request)
            return len(tokens.tokens)
        except Exception:
            total = sum(len(m.get("content", "")) for m in messages)
            return max(1, total // 4)
