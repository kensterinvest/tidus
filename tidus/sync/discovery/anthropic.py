"""Anthropic model discovery.

Anthropic's `/v1/models` endpoint differs from the OpenAI shape:
  * Auth via `x-api-key` header (not `Authorization: Bearer`)
  * Requires `anthropic-version` header
  * Items carry `id`, `display_name`, `created_at`, `type="model"`
  * IDs include date suffixes (e.g. `claude-opus-4-7-20260420`); we strip
    those for canonical matching against Tidus' registry IDs
    (`claude-opus-4-7`).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import httpx
import structlog

from tidus.sync.discovery.base import DiscoveredModel, DiscoverySource

log = structlog.get_logger(__name__)

# Anthropic appends `-YYYYMMDD` to model IDs for stable references
# (e.g. claude-sonnet-4-6-20260101). Tidus uses unsuffixed canonicals.
_DATE_SUFFIX = re.compile(r"-\d{8}$")
_API_VERSION = "2023-06-01"


def _canonicalize_anthropic(raw_id: str) -> str:
    return _DATE_SUFFIX.sub("", raw_id)


class AnthropicDiscoverySource(DiscoverySource):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout_seconds: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def source_name(self) -> str:
        return "anthropic-models"

    @property
    def vendor(self) -> str:
        return "anthropic"

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    async def list_models(self) -> list[DiscoveredModel]:
        url = f"{self._base_url}/v1/models?limit=200"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            log.warning(
                "discovery_fetch_failed",
                source=self.source_name,
                vendor=self.vendor,
                error=str(exc),
            )
            return []
        except ValueError as exc:
            log.warning(
                "discovery_parse_failed",
                source=self.source_name,
                vendor=self.vendor,
                error=str(exc),
            )
            return []

        items = payload.get("data") or []
        retrieved_at = datetime.now(UTC)
        out: list[DiscoveredModel] = []

        for item in items:
            raw_id = (item.get("id") or "").strip()
            if not raw_id:
                continue
            canonical_id = _canonicalize_anthropic(raw_id)
            out.append(
                DiscoveredModel(
                    model_id=canonical_id,
                    vendor_id=raw_id,
                    vendor=self.vendor,
                    display_name=item.get("display_name"),
                    source_name=self.source_name,
                    retrieved_at=retrieved_at,
                    raw_metadata={
                        k: v for k, v in item.items()
                        if k in ("created_at", "type")
                    },
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
