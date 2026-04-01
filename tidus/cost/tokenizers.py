"""Per-vendor token counting dispatch.

Each vendor family uses a different tokenizer. This module provides a single
async entry point — count_tokens(model, messages) — that dispatches to the
correct implementation based on model.tokenizer.

Tiktoken (OpenAI/DeepSeek/Grok/Kimi): fast, local, no network.
Anthropic: uses the count_tokens API endpoint (network call).
Google: uses the generativeai SDK count_tokens method.
Sentencepiece (Mistral): local sentencepiece model file.
Ollama: calls /api/tokenize on the local Ollama server.

Example:
    n = await count_tokens(spec, messages)
"""

from __future__ import annotations

import json

import httpx

from tidus.models.model_registry import ModelSpec, TokenizerType
from tidus.settings import get_settings

# Lazy imports — only loaded when the relevant tokenizer is first used
_tiktoken_encoders: dict[str, object] = {}
_anthropic_client: object | None = None
_google_models: dict[str, object] = {}
_mistral_tokenizer: object | None = None


def _flatten_messages(messages: list[dict]) -> str:
    """Concatenate message contents into a single string for tokenizers that
    require a flat prompt (Mistral sentencepiece, Ollama)."""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-modal content: extract text parts only
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


async def count_tokens(model: ModelSpec, messages: list[dict]) -> int:
    """Dispatch to the correct tokenizer for the given model.

    Returns the estimated number of input tokens. For tiktoken-based models,
    adds 3 tokens per message for the chat format overhead (role + delimiters).

    Raises RuntimeError if the tokenizer library is not installed or
    the remote tokenize endpoint is unreachable.
    """
    match model.tokenizer:
        case TokenizerType.tiktoken_cl100k | TokenizerType.tiktoken_o200k:
            return _count_tiktoken(model.tokenizer, messages)
        case TokenizerType.anthropic:
            return await _count_anthropic(model.model_id, messages)
        case TokenizerType.google:
            return _count_google(model.model_id, messages)
        case TokenizerType.sentencepiece:
            return _count_sentencepiece(messages)
        case TokenizerType.ollama:
            return await _count_ollama(model.model_id, messages)
        case _:
            raise RuntimeError(f"Unknown tokenizer type: {model.tokenizer}")


# ── Tiktoken (OpenAI / DeepSeek / Grok / Kimi) ───────────────────────────────

def _count_tiktoken(tokenizer_type: TokenizerType, messages: list[dict]) -> int:
    try:
        import tiktoken  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("tiktoken not installed: pip install tiktoken") from exc

    enc_name = tokenizer_type.value  # "tiktoken_cl100k" or "tiktoken_o200k"
    # Map our internal names to tiktoken encoding names
    encoding_map = {
        "tiktoken_cl100k": "cl100k_base",
        "tiktoken_o200k": "o200k_base",
    }
    enc_key = encoding_map.get(enc_name, "cl100k_base")

    if enc_key not in _tiktoken_encoders:
        _tiktoken_encoders[enc_key] = tiktoken.get_encoding(enc_key)

    enc = _tiktoken_encoders[enc_key]
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        total += len(enc.encode(content)) + 3  # role + delimiters overhead
    return total


# ── Anthropic ─────────────────────────────────────────────────────────────────

async def _count_anthropic(model_id: str, messages: list[dict]) -> int:
    global _anthropic_client
    try:
        import anthropic  # type: ignore[import-untyped]
    except ImportError:
        return _fallback_count(messages)

    settings = get_settings()
    if not settings.anthropic_api_key or settings.anthropic_api_key.startswith("sk-ant-..."):
        return _fallback_count(messages)

    try:
        if _anthropic_client is None:
            _anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        response = await _anthropic_client.messages.count_tokens(
            model=model_id,
            messages=messages,
        )
        return response.input_tokens
    except Exception:
        # Auth failure, network error, or quota — fall back to local approximation
        return _fallback_count(messages)


# ── Google ────────────────────────────────────────────────────────────────────

def _count_google(model_id: str, messages: list[dict]) -> int:
    try:
        import google.generativeai as genai  # type: ignore[import-untyped]
    except ImportError:
        return _fallback_count(messages)

    settings = get_settings()
    if not settings.google_api_key or settings.google_api_key.startswith("AIza..."):
        return _fallback_count(messages)

    try:
        genai.configure(api_key=settings.google_api_key)
        if model_id not in _google_models:
            _google_models[model_id] = genai.GenerativeModel(model_id)
        gm = _google_models[model_id]
        parts = [{"role": m.get("role", "user"), "parts": [m.get("content", "")]} for m in messages]
        result = gm.count_tokens(parts)
        return result.total_tokens
    except Exception:
        return _fallback_count(messages)


# ── Sentencepiece (Mistral) ───────────────────────────────────────────────────

def _count_sentencepiece(messages: list[dict]) -> int:
    global _mistral_tokenizer
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # type: ignore[import-untyped]
        from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage, SystemMessage  # type: ignore[import-untyped]
        from mistral_common.protocol.instruct.request import ChatCompletionRequest  # type: ignore[import-untyped]
    except ImportError:
        # Fall back to approximate count if mistral_common is not available
        flat = _flatten_messages(messages)
        # Rough approximation: ~4 chars per token (similar to other models)
        return max(1, len(flat) // 4)

    if _mistral_tokenizer is None:
        _mistral_tokenizer = MistralTokenizer.v3()

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
    tokens = _mistral_tokenizer.encode_chat_completion(request)
    return len(tokens.tokens)


# ── Ollama ────────────────────────────────────────────────────────────────────

async def _count_ollama(model_id: str, messages: list[dict]) -> int:
    settings = get_settings()
    base_url = settings.ollama_base_url.rstrip("/")
    prompt = _flatten_messages(messages)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{base_url}/api/tokenize",
                json={"model": model_id, "prompt": prompt},
            )
            resp.raise_for_status()
            data = resp.json()
            tokens: list[int] = data.get("tokens", [])
            return len(tokens)
    except Exception:
        # Ollama not running or unreachable — fall back to local approximation
        return _fallback_count(messages)


# ── Fallback ──────────────────────────────────────────────────────────────────

def _fallback_count(messages: list[dict]) -> int:
    """Local token count approximation using tiktoken cl100k_base.

    Used when the vendor tokenizer is unavailable (no API key, network error,
    or SDK not installed). Accurate enough for cost estimation within the
    existing 15% safety buffer.
    """
    try:
        import tiktoken  # type: ignore[import-untyped]
        if "cl100k_base" not in _tiktoken_encoders:
            _tiktoken_encoders["cl100k_base"] = tiktoken.get_encoding("cl100k_base")
        enc = _tiktoken_encoders["cl100k_base"]
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            total += len(enc.encode(content)) + 3
        return total
    except Exception:
        # Last resort: character-based approximation (~4 chars per token)
        flat = _flatten_messages(messages)
        return max(1, len(flat) // 4)
