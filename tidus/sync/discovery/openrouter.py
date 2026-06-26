"""OpenRouterDiscoverySource — vendor-agnostic catalog discovery.

The official vendor `/v1/models` endpoints (one per vendor source under
this package) require per-vendor API keys. OpenRouter's public
`/api/v1/models` endpoint is unauthenticated and lists every model
brokered across all major vendors in a single response. That means it
can surface a brand-new Google or Anthropic model the moment OpenRouter
indexes it — without any GitHub secret configuration.

This source is COMPLEMENTARY to the per-vendor discovery sources:
  * If a vendor API key IS configured, that vendor's first-party source
    is preferred (more authoritative, lower latency surface).
  * If no key is configured, OpenRouter still keeps the magazine fresh.
  * When both fire, the runner deduplicates by canonical model_id
    (first-seen wins, and per-vendor sources are registered first in
    factory.py — see ordering note there).

Pricing is captured into `raw_metadata` so the magazine can render
"new from $X/1M" alongside pending-review entries even before someone
adds the model to config/models.yaml.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import structlog

from tidus.sync.discovery.base import DiscoveredModel, DiscoverySource
from tidus.sync.openrouter_id_map import canonical_from_openrouter, strip_variant

log = structlog.get_logger(__name__)

# Vendor prefix → canonical Tidus vendor key. Matches the keys used in
# tidus/reporting/landing_updater.py _VENDOR_NAMES. Anything not listed
# falls back to the raw prefix string (the magazine title-cases it).
_VENDOR_PREFIX_MAP: dict[str, str] = {
    "openai":      "openai",
    "anthropic":   "anthropic",
    "google":      "google",
    "mistralai":   "mistral",
    "deepseek":    "deepseek",
    "x-ai":        "xai",
    "moonshotai":  "moonshot",
    "cohere":      "cohere",
    "qwen":        "qwen",
    "perplexity":  "perplexity",
    "meta-llama":  "meta",
    "alibaba":     "alibaba",
    "z-ai":        "zhipu",
}


def _vendor_from_or_id(or_id: str) -> str:
    """Extract the canonical vendor key from an OpenRouter id `vendor/model`."""
    if "/" not in or_id:
        return ""
    prefix = or_id.split("/", 1)[0]
    return _VENDOR_PREFIX_MAP.get(prefix, prefix)


# Re-export for backwards compatibility with existing tests.
_strip_variant = strip_variant


class OpenRouterDiscoverySource(DiscoverySource):
    """Surfaces every model OpenRouter brokers, no API key required.

    The DiscoveryRunner diffs the output against the current Tidus
    registry to decide which discovered models are new vs already known
    vs pending review.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://openrouter.ai",
        timeout_seconds: float = 15.0,
        enabled: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._enabled = enabled

    @property
    def source_name(self) -> str:
        return "openrouter-discovery"

    @property
    def vendor(self) -> str:
        # OpenRouter is multi-vendor — use a sentinel rather than picking one.
        return "openrouter"

    @property
    def is_available(self) -> bool:
        return self._enabled

    async def list_models(self) -> list[DiscoveredModel]:
        if not self._enabled:
            return []

        url = f"{self._base_url}/api/v1/models"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            log.warning("discovery_fetch_failed", source=self.source_name, error=str(exc))
            return []
        except ValueError as exc:
            log.warning("discovery_parse_failed", source=self.source_name, error=str(exc))
            return []

        items = payload.get("data") or []
        retrieved_at = datetime.now(UTC)

        out: list[DiscoveredModel] = []
        seen_canonical: set[str] = set()

        for item in items:
            raw_id = (item.get("id") or "").strip()
            if not raw_id or "/" not in raw_id:
                continue

            # Use the shared OpenRouter → Tidus canonical map. Plain
            # slash-strip produces dot-versioned Anthropic ids
            # (`claude-opus-4.6`) that duplicate hand-curated dash ids
            # (`claude-opus-4-6`) and would 404 if routed. The shared
            # map normalizes both sources to the same answer.
            canonical = canonical_from_openrouter(raw_id)
            if not canonical:
                continue

            # First-write wins on duplicate canonicals (OpenRouter sometimes
            # lists the same model under multiple variant tags that resolve
            # to the same Tidus canonical id).
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)

            vendor = _vendor_from_or_id(raw_id)

            # Capture pricing into raw_metadata so the magazine renderer can
            # show "$X/1M" alongside a pending-review row. Stored as the raw
            # per-token strings OpenRouter returned — let downstream code
            # decide units to avoid lossy conversions.
            pricing = item.get("pricing") or {}
            metadata = {
                "openrouter_id":   raw_id,
                "context_length":  item.get("context_length"),
                "input_modalities": (item.get("architecture") or {}).get("input_modalities"),
                "pricing": {
                    "prompt":     pricing.get("prompt"),
                    "completion": pricing.get("completion"),
                },
            }

            out.append(
                DiscoveredModel(
                    model_id=canonical,
                    vendor_id=raw_id,
                    vendor=vendor,
                    display_name=item.get("name"),
                    source_name=self.source_name,
                    retrieved_at=retrieved_at,
                    raw_metadata=metadata,
                )
            )

        log.info(
            "discovery_fetched",
            source=self.source_name,
            vendor=self.vendor,
            raw_count=len(items),
            kept_count=len(out),
        )
        return out
