"""Claude web-search model discovery — an intelligence DiscoverySource.

Once per sync, asks Claude (Sonnet 5 + web_search) what launched or changed
in the LLM market since the last sync, returning candidates WITH web-sourced
pricing in raw_metadata. Fail-open: no client / error => [].
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import structlog

from tidus.sync.anthropic_client import SyncTokenLedger
from tidus.sync.discovery.base import DiscoveredModel, DiscoverySource

log = structlog.get_logger(__name__)

_SYSTEM = (
    "You are a market analyst for the Tidus AI model router. Use web search to "
    "find LLM models that launched or materially changed pricing/capability "
    "recently. Only report models you can corroborate from a cited source. "
    "Output strict JSON matching the schema — prices in USD per 1M tokens."
)

_SCHEMA = {
    "type": "object",
    "properties": {"models": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "model_id": {"type": "string"}, "vendor": {"type": "string"},
            "input_usd_per_1m": {"type": "number"}, "output_usd_per_1m": {"type": "number"},
            "purpose": {"type": "string"}, "positioning": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["model_id", "vendor", "input_usd_per_1m", "output_usd_per_1m",
                     "purpose", "positioning", "sources"],
        "additionalProperties": False,
    }}},
    "required": ["models"],
    "additionalProperties": False,
}


class ClaudeMarketDiscoverySource(DiscoverySource):
    """Asks Claude (with web search) what's new in the LLM market.

    Unlike the vendor-catalog sources in this package, this one is
    multi-vendor and speculative — candidates come with Claude's own
    web-sourced pricing rather than a verified vendor API response, so
    they're clearly tagged (`claude_sourced=True`) for the promotion
    review to weigh accordingly.
    """

    def __init__(self, *, client, ledger: SyncTokenLedger, model: str, last_sync_date: str) -> None:
        self._client = client
        self._ledger = ledger
        self._model = model
        self._last_sync_date = last_sync_date

    @property
    def source_name(self) -> str:
        return "claude-market"

    @property
    def vendor(self) -> str:
        return "*"  # multi-vendor

    @property
    def is_available(self) -> bool:
        return self._client is not None

    async def list_models(self) -> list[DiscoveredModel]:
        if self._client is None:
            return []
        prompt = (
            f"Since {self._last_sync_date}, which LLM models launched or changed "
            "pricing/capability? For each, give canonical model_id, vendor, "
            "input/output USD per 1M tokens, a one-line purpose, positioning "
            "(flagship/mid/economy), and source URLs."
        )
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=4096,
                system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=[{"type": "web_search_20260209", "name": "web_search"}],
                output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 — fail-open by design
            log.warning("claude_market_discovery_failed", error=str(exc))
            return []

        content = getattr(resp, "content", []) or []
        searches = sum(1 for b in content if getattr(b, "type", "") == "server_tool_use")
        self._ledger.record("discovery", getattr(resp, "usage", None), web_searches=searches)

        text = next((b.text for b in content if getattr(b, "type", "") == "text"), "")
        try:
            payload = json.loads(text) if text else {"models": []}
        except json.JSONDecodeError as exc:
            log.warning("claude_market_parse_failed", error=str(exc))
            return []

        now = datetime.now(UTC)
        out: list[DiscoveredModel] = []
        for m in payload.get("models", []):
            try:
                out.append(DiscoveredModel(
                    model_id=str(m["model_id"]).strip(),
                    vendor_id=str(m["model_id"]).strip(),
                    vendor=str(m["vendor"]).strip(),
                    display_name=m.get("model_id"),
                    source_name=self.source_name,
                    retrieved_at=now,
                    raw_metadata={
                        "claude_sourced": True,
                        "price_in_per_1k": round(float(m["input_usd_per_1m"]) / 1000, 6),
                        "price_out_per_1k": round(float(m["output_usd_per_1m"]) / 1000, 6),
                        "purpose": m.get("purpose", ""),
                        "positioning": m.get("positioning", ""),
                        "sources": list(m.get("sources", [])),
                    },
                ))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("claude_market_candidate_dropped", error=str(exc), raw=str(m)[:120])
        log.info("claude_market_discovered", count=len(out))
        return out
