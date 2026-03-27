"""Ollama adapter — local model inference via the Ollama REST API.

No API key required. Connects to a locally running Ollama server.
Supports all models available in the local Ollama installation.

Environment:
    OLLAMA_BASE_URL: URL of the Ollama server (default: http://localhost:11434)
"""

from __future__ import annotations

import time
from typing import AsyncIterator

import httpx
import structlog

from tidus.adapters.base import AbstractModelAdapter, AdapterResponse, register_adapter
from tidus.settings import get_settings

log = structlog.get_logger(__name__)

_PROBE_PROMPT = "hi"


@register_adapter
class OllamaAdapter(AbstractModelAdapter):
    vendor = "ollama"

    async def complete(self, model_id: str, task) -> AdapterResponse:
        settings = get_settings()
        base_url = settings.ollama_base_url.rstrip("/")

        payload = {
            "model": model_id,
            "messages": task.messages,
            "stream": False,
            "options": {"num_predict": task.estimated_output_tokens},
        }

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()

        latency_ms = (time.monotonic() - start) * 1000
        data = resp.json()

        message = data.get("message", {})
        content = message.get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        log.info(
            "ollama_complete",
            model_id=model_id,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            latency_ms=round(latency_ms, 1),
        )
        return AdapterResponse(
            model_id=model_id,
            content=content,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            latency_ms=latency_ms,
            finish_reason="stop",
            raw=data,
        )

    async def stream_complete(self, model_id: str, task) -> AsyncIterator[str]:
        settings = get_settings()
        base_url = settings.ollama_base_url.rstrip("/")

        payload = {
            "model": model_id,
            "messages": task.messages,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break

    async def health_check(self, model_id: str) -> bool:
        settings = get_settings()
        base_url = settings.ollama_base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{base_url}/api/chat",
                    json={"model": model_id, "messages": [{"role": "user", "content": _PROBE_PROMPT}], "stream": False},
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def count_tokens(self, model_id: str, messages: list[dict]) -> int:
        settings = get_settings()
        base_url = settings.ollama_base_url.rstrip("/")
        prompt = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{base_url}/api/tokenize",
                    json={"model": model_id, "prompt": prompt},
                )
                resp.raise_for_status()
                return len(resp.json().get("tokens", []))
        except Exception:
            return max(1, len(prompt) // 4)
