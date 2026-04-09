"""HardcodedSource — wraps the built-in verified-prices dict.

Always available (is_available=True). Confidence=0.7 reflects that prices
are manually maintained and may lag actual vendor changes by days-to-weeks.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from tidus.sync.pricing.base import PriceQuote, PricingSource

# Prices verified: 2026-04-09. Updated by host's sync_pricing.py script.
# Source: official vendor pricing pages (direct API prices, not OpenRouter).
# All prices in $/1K tokens. $/1M = value * 1000.
_KNOWN_PRICES: dict[str, dict[str, float]] = {
    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-haiku-4-5":    {"input": 0.0008,   "output": 0.004},   # $0.80/$4 per 1M
    "claude-opus-4-6":     {"input": 0.015,    "output": 0.075},   # $15/$75 per 1M
    "claude-sonnet-4-6":   {"input": 0.003,    "output": 0.015},   # $3/$15 per 1M

    # ── DeepSeek ─────────────────────────────────────────────────────────────
    "deepseek-r1":         {"input": 0.00055,  "output": 0.00219}, # $0.55/$2.19 per 1M (non-cached)
    "deepseek-v3":         {"input": 0.00027,  "output": 0.00089}, # $0.27/$0.89 per 1M (updated Apr 2026)

    # ── Google ────────────────────────────────────────────────────────────────
    "gemini-2.0-flash":    {"input": 0.0001,   "output": 0.0004},  # $0.10/$0.40 per 1M
    "gemini-2.5-flash":    {"input": 0.0003,   "output": 0.0025},  # $0.30/$2.50 per 1M
    "gemini-2.5-pro":      {"input": 0.00125,  "output": 0.01},    # $1.25/$10 per 1M (<=200K ctx)

    # ── OpenAI ───────────────────────────────────────────────────────────────
    "gpt-4.1":             {"input": 0.002,    "output": 0.008},   # $2/$8 per 1M
    "gpt-4.1-mini":        {"input": 0.0004,   "output": 0.0016},  # $0.40/$1.60 per 1M
    "gpt-4.1-nano":        {"input": 0.0001,   "output": 0.0004},  # $0.10/$0.40 per 1M
    "gpt-4o":              {"input": 0.0025,   "output": 0.01},    # $2.50/$10 per 1M
    "gpt-4o-mini":         {"input": 0.00015,  "output": 0.0006},  # $0.15/$0.60 per 1M
    "gpt-5-codex":         {"input": 0.00125,  "output": 0.01},    # $1.25/$10 per 1M
    "o3":                  {"input": 0.010,    "output": 0.040},   # $10/$40 per 1M (reasoning model)
    "o4-mini":             {"input": 0.0011,   "output": 0.0044},  # $1.10/$4.40 per 1M

    # ── xAI ──────────────────────────────────────────────────────────────────
    "grok-3":              {"input": 0.003,    "output": 0.015},   # $3/$15 per 1M
    "grok-3-fast":         {"input": 0.005,    "output": 0.025},   # $5/$25 per 1M

    # ── Qwen (disabled in YAML but prices tracked for re-enable) ─────────────
    "qwen-max":            {"input": 0.00104,  "output": 0.00416}, # $1.04/$4.16 per 1M
    "qwen-plus":           {"input": 0.00026,  "output": 0.00078}, # $0.26/$0.78 per 1M

    # ── Perplexity (disabled in YAML but prices tracked) ─────────────────────
    "sonar":               {"input": 0.001,    "output": 0.001},   # $1/$1 per 1M
    "sonar-pro":           {"input": 0.003,    "output": 0.015},   # $3/$15 per 1M
}

_EFFECTIVE_DATE = date(2026, 4, 9)
_CONFIDENCE = 0.7


class HardcodedSource(PricingSource):
    """Built-in price table — always available, confidence=0.7."""

    @property
    def source_name(self) -> str:
        return "hardcoded"

    @property
    def confidence(self) -> float:
        return _CONFIDENCE

    async def fetch_quotes(self) -> list[PriceQuote]:
        now = datetime.now(UTC)
        return [
            PriceQuote(
                model_id=model_id,
                input_price=prices["input"],
                output_price=prices["output"],
                cache_read_price=0.0,
                cache_write_price=0.0,
                currency="USD",
                effective_date=_EFFECTIVE_DATE,
                retrieved_at=now,
                source_name=self.source_name,
                source_confidence=_CONFIDENCE,
            )
            for model_id, prices in _KNOWN_PRICES.items()
        ]
