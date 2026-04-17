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
    # ── OpenAI ───────────────────────────────────────────────────────────────
    "o3":                        {"input": 0.002,     "output": 0.008},    # $2/$8 per 1M (updated Apr-17)
    "o4-mini":                   {"input": 0.0011,    "output": 0.0044},   # $1.10/$4.40 per 1M
    "gpt-4.1":                   {"input": 0.002,     "output": 0.008},    # $2/$8 per 1M
    "gpt-4.1-mini":              {"input": 0.0004,    "output": 0.0016},   # $0.40/$1.60 per 1M
    "gpt-4.1-nano":              {"input": 0.0001,    "output": 0.0004},   # $0.10/$0.40 per 1M
    "gpt-4o":                    {"input": 0.0025,    "output": 0.01},     # $2.50/$10 per 1M
    "gpt-4o-mini":               {"input": 0.00015,   "output": 0.0006},   # $0.15/$0.60 per 1M
    "gpt-oss-120b":              {"input": 0.000039,  "output": 0.0001},   # $0.039/$0.10 per 1M
    "gpt-5-codex":               {"input": 0.00125,   "output": 0.01},     # $1.25/$10 per 1M
    "codex-mini-latest":         {"input": 0.00075,   "output": 0.003},    # $0.75/$3 per 1M

    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-opus-4-7":           {"input": 0.005,     "output": 0.025},    # $5/$25 per 1M (new Apr-17)
    "claude-opus-4-6":           {"input": 0.005,     "output": 0.025},    # $5/$25 per 1M (updated Apr-17)
    "claude-sonnet-4-6":         {"input": 0.003,     "output": 0.015},    # $3/$15 per 1M
    "claude-haiku-4-5":          {"input": 0.001,     "output": 0.005},    # $1/$5 per 1M (updated Apr-17)

    # ── Google ────────────────────────────────────────────────────────────────
    "gemini-3.1-pro":            {"input": 0.002,     "output": 0.012},    # $2/$12 per 1M
    "gemini-3.1-flash":          {"input": 0.001,     "output": 0.004},    # $1/$4 per 1M
    "gemini-2.5-pro":            {"input": 0.00125,   "output": 0.01},     # $1.25/$10 per 1M
    "gemini-2.5-flash":          {"input": 0.0003,    "output": 0.0025},   # $0.30/$2.50 per 1M
    "gemini-2.0-flash":          {"input": 0.0001,    "output": 0.0004},   # $0.10/$0.40 per 1M

    # ── Mistral ───────────────────────────────────────────────────────────────
    "mistral-large-3":           {"input": 0.002,     "output": 0.006},    # $2/$6 per 1M
    "mistral-medium":            {"input": 0.0004,    "output": 0.002},    # $0.40/$2 per 1M
    "mistral-small":             {"input": 0.0001,    "output": 0.0003},   # $0.10/$0.30 per 1M
    "mistral-nemo":              {"input": 0.00015,   "output": 0.00015},  # $0.15/$0.15 per 1M
    "codestral":                 {"input": 0.0002,    "output": 0.0006},   # $0.20/$0.60 per 1M
    "devstral":                  {"input": 0.0004,    "output": 0.002},    # $0.40/$2 per 1M
    "devstral-small":            {"input": 0.0001,    "output": 0.0003},   # $0.10/$0.30 per 1M

    # ── DeepSeek ─────────────────────────────────────────────────────────────
    "deepseek-r1":               {"input": 0.0007,    "output": 0.0025},   # $0.70/$2.50 per 1M (updated Apr-17)
    "deepseek-v3":               {"input": 0.00032,   "output": 0.00089},  # $0.32/$0.89 per 1M (updated Apr-17)
    "deepseek-v4":               {"input": 0.0003,    "output": 0.0005},   # $0.30/$0.50 per 1M

    # ── xAI ──────────────────────────────────────────────────────────────────
    "grok-4":                    {"input": 0.003,     "output": 0.015},    # $3/$15 per 1M (released Jul-2025)
    "grok-3":                    {"input": 0.003,     "output": 0.015},    # $3/$15 per 1M
    "grok-3-fast":               {"input": 0.005,     "output": 0.025},    # $5/$25 per 1M

    # ── Moonshot / Kimi ───────────────────────────────────────────────────────
    "kimi-k2.5":                 {"input": 0.0006,    "output": 0.0025},   # $0.60/$2.50 per 1M

    # ── Cohere (disabled in YAML but prices tracked) ──────────────────────────
    "command-r-plus":            {"input": 0.0025,    "output": 0.01},     # $2.50/$10 per 1M
    "command-r":                 {"input": 0.00015,   "output": 0.0006},   # $0.15/$0.60 per 1M

    # ── Qwen (partially disabled in YAML but prices tracked) ─────────────────
    "qwen-max":                  {"input": 0.00104,   "output": 0.00416},  # $1.04/$4.16 per 1M
    "qwen-plus":                 {"input": 0.00026,   "output": 0.00078},  # $0.26/$0.78 per 1M
    "qwen-flash":                {"input": 0.0004,    "output": 0.0012},   # $0.40/$1.20 per 1M

    # ── Perplexity (disabled in YAML but prices tracked) ─────────────────────
    "sonar-pro":                 {"input": 0.003,     "output": 0.015},    # $3/$15 per 1M
    "sonar":                     {"input": 0.001,     "output": 0.001},    # $1/$1 per 1M

    # ── Groq (hosted inference) ───────────────────────────────────────────────
    "groq-llama4-maverick":      {"input": 0.0005,    "output": 0.0015},   # $0.50/$1.50 per 1M
    "groq-deepseek-r1":          {"input": 0.001,     "output": 0.003},    # $1/$3 per 1M

    # ── Together AI (hosted inference) ───────────────────────────────────────
    "together-llama4-maverick":  {"input": 0.00027,   "output": 0.00085},  # $0.27/$0.85 per 1M
}

_EFFECTIVE_DATE = date(2026, 4, 17)
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
