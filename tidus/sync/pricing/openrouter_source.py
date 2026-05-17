"""OpenRouterPricingSource — live vendor pricing via openrouter.ai.

OpenRouter's `/api/v1/models` endpoint is unauthenticated and lists every
model the platform brokers, complete with current per-token pricing. That
makes it the cheapest way to get "second-opinion" pricing for every major
vendor in one HTTP call.

Why this source exists:
  HardcodedSource alone means the pipeline compares the same numbers to
  themselves on every Sun+Wed run, so the magazine never moves between
  manual price edits. Adding OpenRouter gives the consensus layer a live
  feed to compare against — when a vendor cuts prices, the change shows up
  on the next sync without a human in the loop.

Response shape (excerpt):
    {
      "data": [
        {
          "id": "google/gemini-2.5-pro",
          "pricing": {
            "prompt":     "0.00000125",   // USD per *token*
            "completion": "0.000005",
            "input_cache_read":  "0.0",   // optional
            "input_cache_write": "0.0"    // optional
          },
          ...
        }
      ]
    }

Mapping:
  OpenRouter id format is `<vendor>/<model>[:variant]`. Tidus canonical ids
  drop the vendor prefix. A small explicit override table handles the
  cases where the slash-stripped suffix doesn't equal the Tidus id (for
  example `anthropic/claude-sonnet-4` vs Tidus `claude-sonnet-4-6`).
  Models that don't match are silently skipped — consensus.py handles
  the empty-quote case cleanly.

Confidence:
  0.75 — higher than HardcodedSource (0.70) because the data is live, but
  not so high that a single bad payload could flip an established price by
  itself (the MAD outlier check in consensus.py still applies).

Failure modes:
  Network errors, HTTP errors, malformed payload → return []. The pipeline
  treats this source as just absent for that run; HardcodedSource keeps
  the registry stable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from tidus.sync.openrouter_id_map import (
    OPENROUTER_TO_TIDUS,
    canonical_from_openrouter,
    strip_variant,
)
from tidus.sync.pricing.base import PriceQuote, PricingSource

log = structlog.get_logger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai"
_DEFAULT_TIMEOUT = 15.0
_CONFIDENCE = 0.75
_TOKENS_PER_THOUSAND = 1000.0  # OpenRouter quotes per token; Tidus stores per 1K

# Re-exports — preserved for backwards compatibility with existing test imports.
_OPENROUTER_TO_TIDUS = OPENROUTER_TO_TIDUS
_strip_variant = strip_variant
_canonical_from_openrouter = canonical_from_openrouter

def _parse_price(raw: Any) -> float:
    """OpenRouter encodes prices as strings — sometimes scientific notation.
    Return 0.0 for missing/None/unparseable, which matches Tidus's "no price"
    convention for fields a model doesn't actually charge for (cache, etc.)."""
    if raw is None or raw == "":
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


class OpenRouterPricingSource(PricingSource):
    """Live multi-vendor pricing via OpenRouter's public catalog endpoint."""

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
        enabled: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._enabled = enabled

    @property
    def source_name(self) -> str:
        return "openrouter"

    @property
    def confidence(self) -> float:
        return _CONFIDENCE

    @property
    def is_available(self) -> bool:
        return self._enabled

    async def fetch_quotes(self) -> list[PriceQuote]:
        if not self._enabled:
            return []

        url = f"{self._base_url}/api/v1/models"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            log.warning("openrouter_fetch_failed", url=url, error=str(exc))
            return []
        except ValueError as exc:
            log.warning("openrouter_parse_failed", url=url, error=str(exc))
            return []

        items = payload.get("data") or []
        now = datetime.now(UTC)
        today = now.date()

        quotes: list[PriceQuote] = []
        seen_canonical: set[str] = set()

        for item in items:
            or_id = (item.get("id") or "").strip()
            if not or_id:
                continue

            canonical = _canonical_from_openrouter(or_id)
            if not canonical:
                continue

            # OpenRouter occasionally lists multiple variants of the same
            # model (e.g. provider duplicates). First write wins — list order
            # tends to be highest-quality first.
            if canonical in seen_canonical:
                continue

            pricing = item.get("pricing") or {}
            prompt_per_token = _parse_price(pricing.get("prompt"))
            completion_per_token = _parse_price(pricing.get("completion"))
            if prompt_per_token <= 0 and completion_per_token <= 0:
                # Free / unpriced model — nothing useful for consensus.
                continue

            cache_read_per_token = _parse_price(pricing.get("input_cache_read"))
            cache_write_per_token = _parse_price(pricing.get("input_cache_write"))

            seen_canonical.add(canonical)
            quotes.append(
                PriceQuote(
                    model_id=canonical,
                    input_price=prompt_per_token * _TOKENS_PER_THOUSAND,
                    output_price=completion_per_token * _TOKENS_PER_THOUSAND,
                    cache_read_price=cache_read_per_token * _TOKENS_PER_THOUSAND,
                    cache_write_price=cache_write_per_token * _TOKENS_PER_THOUSAND,
                    currency="USD",
                    effective_date=today,
                    retrieved_at=now,
                    source_name=self.source_name,
                    source_confidence=_CONFIDENCE,
                    evidence_url=f"{self._base_url}/models/{or_id}",
                )
            )

        log.info(
            "openrouter_fetched",
            raw_count=len(items),
            mapped_count=len(quotes),
        )
        return quotes
