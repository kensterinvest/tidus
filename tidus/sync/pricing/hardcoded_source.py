"""HardcodedSource — wraps the built-in verified-prices dict.

Always available (is_available=True). Confidence=0.7 reflects that prices
are manually maintained and may lag actual vendor changes by days-to-weeks.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from tidus.sync.pricing.base import PriceQuote, PricingSource

# Prices verified: 2026-04-17. Updated by host's sync_pricing.py script.
# Source: official vendor pricing pages (direct API prices, not OpenRouter).
# All prices in $/1K tokens. $/1M = value * 1000.
_KNOWN_PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":     {"input": 0.001, "output": 0.005},
    "claude-opus-4-6":      {"input": 0.005, "output": 0.025},
    "claude-opus-4-7":      {"input": 0.005, "output": 0.025},
    "claude-sonnet-4-6":    {"input": 0.003, "output": 0.015},
    "deepseek-r1":          {"input": 0.0007, "output": 0.0025},
    "deepseek-v3":          {"input": 0.00032, "output": 0.00089},
    "gemini-2.5-flash":     {"input": 0.0003, "output": 0.0025},
    "gemini-2.5-pro":       {"input": 0.00125, "output": 0.01},
    "gpt-4.1":              {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini":         {"input": 0.0004, "output": 0.0016},
    "gpt-4.1-nano":         {"input": 0.0001, "output": 0.0004},
    "gpt-4o":               {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini":          {"input": 0.00015, "output": 0.0006},
    "gpt-5-codex":          {"input": 0.00125, "output": 0.01},
    "grok-3":               {"input": 0.003, "output": 0.015},
    "o3":                   {"input": 0.002, "output": 0.008},
    "o4-mini":              {"input": 0.0011, "output": 0.0044},
    "qwen-max":             {"input": 0.00104, "output": 0.00416},
    "qwen-plus":            {"input": 0.00026, "output": 0.00078},
    "sonar":                {"input": 0.001, "output": 0.001},
    "sonar-pro":            {"input": 0.003, "output": 0.015},
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
