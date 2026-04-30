"""Build the list of DiscoverySources to run, based on which vendor API
keys are configured. Sources without keys are excluded (not just skipped)
so the runner's `sources_skipped` log only shows things the user actually
intended to run.
"""

from __future__ import annotations

from tidus.settings import Settings, get_settings
from tidus.sync.discovery.anthropic import AnthropicDiscoverySource
from tidus.sync.discovery.base import DiscoverySource
from tidus.sync.discovery.openai_compatible import (
    deepseek_source,
    mistral_source,
    openai_source,
    xai_source,
)


def build_discovery_sources(settings: Settings | None = None) -> list[DiscoverySource]:
    """Return the configured discovery sources.

    A vendor is included iff its API key env var is set. The runner will
    further filter on `is_available` (which currently equals key-presence
    but could expand to include circuit breakers later).

    Vendors NOT yet covered:
      - Google: uses generative-ai-python SDK; not OpenAI-compatible
      - Moonshot, Cohere, Groq, Qwen, Together, Perplexity:
        either disabled in routing or use bespoke shapes — add as needed.
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
    if s.mistral_api_key:
        sources.append(mistral_source(s.mistral_api_key))
    if s.deepseek_api_key:
        sources.append(deepseek_source(s.deepseek_api_key))
    if s.xai_api_key:
        sources.append(xai_source(s.xai_api_key))

    return sources
