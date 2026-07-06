"""Sync-only Anthropic client factory + per-run token/cost ledger.

The ONLY place in tidus/sync that constructs an Anthropic client. Reads a
dedicated key (settings.tidus_sync_anthropic_key) so a Claude Code session
or any ambient-credential SDK call can never consume it. Fail-open: no key
=> None => callers skip their Claude pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from tidus.settings import get_settings

log = structlog.get_logger(__name__)

# Sonnet-5 list price, USD per 1M tokens. Verify against pricing.md before go-live.
SONNET_5_INPUT_PER_1M = 3.0
SONNET_5_OUTPUT_PER_1M = 15.0
WEB_SEARCH_PER_CALL = 0.01


@dataclass
class _StageUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    web_searches: int = 0


@dataclass
class SyncTokenLedger:
    """Aggregates Claude usage across a single sync run, tagged by stage."""

    stages: dict[str, _StageUsage] = field(default_factory=dict)

    def record(self, stage: str, usage, web_searches: int = 0) -> None:
        s = self.stages.setdefault(stage, _StageUsage())
        s.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        s.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        s.cache_read_tokens += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        s.cache_creation_tokens += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        s.web_searches += web_searches

    def estimated_usd(self) -> float:
        total_in = sum(
            s.input_tokens + s.cache_read_tokens + s.cache_creation_tokens
            for s in self.stages.values()
        )
        total_out = sum(s.output_tokens for s in self.stages.values())
        searches = sum(s.web_searches for s in self.stages.values())
        return (
            total_in / 1_000_000 * SONNET_5_INPUT_PER_1M
            + total_out / 1_000_000 * SONNET_5_OUTPUT_PER_1M
            + searches * WEB_SEARCH_PER_CALL
        )

    def over_budget(self, ceiling_usd: float) -> bool:
        return self.estimated_usd() > ceiling_usd

    def summary(self) -> dict:
        return {
            "total_input_tokens": sum(s.input_tokens for s in self.stages.values()),
            "total_output_tokens": sum(s.output_tokens for s in self.stages.values()),
            "web_searches": sum(s.web_searches for s in self.stages.values()),
            "estimated_usd": round(self.estimated_usd(), 4),
            "by_stage": {k: vars(v) for k, v in self.stages.items()},
        }


def build_sync_anthropic_client():
    """Return an AsyncAnthropic bound to the sync-only key, or None if unset."""
    key = get_settings().tidus_sync_anthropic_key
    if not key:
        log.info("sync_anthropic_client_unavailable", reason="no_sync_key")
        return None
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        log.warning("sync_anthropic_sdk_missing")
        return None
    return AsyncAnthropic(api_key=key)  # explicit key ONLY — never zero-arg
