"""Build the list of DiscoverySources to run, based on which vendor API
keys are configured. Sources without keys are excluded (not just skipped)
so the runner's `sources_skipped` log only shows things the user actually
intended to run.
"""

from __future__ import annotations

from tidus.settings import Settings, get_settings
from tidus.sync.discovery.anthropic import AnthropicDiscoverySource
from tidus.sync.discovery.base import DiscoverySource
from tidus.sync.discovery.google import GoogleDiscoverySource
from tidus.sync.discovery.openai_compatible import (
    deepseek_source,
    mistral_source,
    openai_source,
    xai_source,
)
from tidus.sync.discovery.openrouter import OpenRouterDiscoverySource


def build_discovery_sources(settings: Settings | None = None) -> list[DiscoverySource]:
    """Return the configured discovery sources.

    Ordering matters: first-listed source wins on duplicate canonical
    model_ids inside the runner's dedup loop. Per-vendor first-party
    sources are registered before OpenRouter so an authoritative quote
    from the vendor itself takes precedence; OpenRouter is the catch-all
    that keeps discovery working when no vendor API keys are configured.

    Vendors with first-party sources: OpenAI, Anthropic, Google, Mistral,
    DeepSeek, xAI. Their entries are included iff the corresponding API
    key env var is set. OpenRouter is included whenever
    `openrouter_enabled` is true (no API key required) and covers
    every vendor it brokers — Moonshot, Cohere, Groq, Qwen, Together,
    Perplexity, Meta, Alibaba, plus the six above.
    """
    s = settings or get_settings()
    timeout = s.discovery_request_timeout_seconds
    sources: list[DiscoverySource] = []

    if s.openai_api_key:
        sources.append(openai_source(s.openai_api_key))
    if s.anthropic_api_key:
        sources.append(
            AnthropicDiscoverySource(
                api_key=s.anthropic_api_key,
                timeout_seconds=timeout,
            )
        )
    if s.google_api_key:
        sources.append(
            GoogleDiscoverySource(
                api_key=s.google_api_key,
                timeout_seconds=timeout,
            )
        )
    if s.mistral_api_key:
        sources.append(mistral_source(s.mistral_api_key))
    if s.deepseek_api_key:
        sources.append(deepseek_source(s.deepseek_api_key))
    if s.xai_api_key:
        sources.append(xai_source(s.xai_api_key))

    if s.openrouter_enabled:
        sources.append(
            OpenRouterDiscoverySource(
                base_url=s.openrouter_base_url,
                timeout_seconds=s.openrouter_request_timeout_seconds,
            )
        )

    return sources
