"""Google AI Studio model discovery.

Uses the Generative Language REST API directly (no SDK):
    GET https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}

Why REST instead of `google.generativeai` or `google.genai`:
  * `google.generativeai` is deprecated (sunset notice as of 2025)
  * `google.genai` is the successor but adds a heavy dependency
  * The REST shape is stable and trivially handled with httpx — same
    pattern as the Anthropic source — so we avoid the SDK entirely
    here even though the routing adapter still uses it.

Response shape:
    {
      "models": [
        {
          "name": "models/gemini-2.5-pro",
          "version": "001",
          "displayName": "Gemini 2.5 Pro",
          "description": "...",
          "inputTokenLimit": 2000000,
          "outputTokenLimit": 8192,
          "supportedGenerationMethods": ["generateContent", "countTokens"]
        },
        ...
      ],
      "nextPageToken": "..."
    }

Filtering rules:
  * Must support `generateContent` (excludes embeddings / countTokens-only / aqa)
  * Excludes any model whose name still contains 'embedding' / 'aqa' as
    a defence-in-depth filter (some embedding models also list
    generateContent on rare occasions)

Canonicalization:
  * Strip leading 'models/' (Google fully-qualifies the resource)
  * Strip trailing '-latest' (Tidus uses unsuffixed canonical IDs)
  * Preview / dated variants (e.g. 'gemini-2.5-pro-preview-05-06') are
    NOT collapsed — they're useful signals that something is coming, and
    a maintainer should decide whether to track them as a separate entry
    or wait for the GA release.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import httpx
import structlog

from tidus.sync.discovery.base import DiscoveredModel, DiscoverySource

log = structlog.get_logger(__name__)

_MODELS_PREFIX = "models/"
_LATEST_SUFFIX = re.compile(r"-latest$")
_DEFAULT_PAGE_SIZE = 200  # Google caps at 1000; 200 is enough for current catalog


def _canonicalize_google(raw_id: str) -> str:
    out = raw_id.removeprefix(_MODELS_PREFIX) if raw_id.startswith(_MODELS_PREFIX) else raw_id
    out = _LATEST_SUFFIX.sub("", out)
    return out


class GoogleDiscoverySource(DiscoverySource):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout_seconds: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def source_name(self) -> str:
        return "google-models"

    @property
    def vendor(self) -> str:
        return "google"

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    async def list_models(self) -> list[DiscoveredModel]:
        """Page through `/v1beta/models` and collect every model that
        supports content generation. Errors degrade to an empty list."""

        retrieved_at = datetime.now(UTC)
        items: list[dict] = []
        page_token: str | None = None

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                while True:
                    params = {"key": self._api_key, "pageSize": _DEFAULT_PAGE_SIZE}
                    if page_token:
                        params["pageToken"] = page_token
                    resp = await client.get(
                        f"{self._base_url}/v1beta/models", params=params
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    items.extend(payload.get("models") or [])
                    page_token = payload.get("nextPageToken")
                    if not page_token:
                        break
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

        out: list[DiscoveredModel] = []
        for item in items:
            raw_id = (item.get("name") or "").strip()
            if not raw_id:
                continue

            methods = item.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            if "embedding" in raw_id.lower() or "aqa" in raw_id.lower():
                continue

            canonical_id = _canonicalize_google(raw_id)
            out.append(
                DiscoveredModel(
                    model_id=canonical_id,
                    vendor_id=raw_id.removeprefix(_MODELS_PREFIX),
                    vendor=self.vendor,
                    display_name=item.get("displayName"),
                    source_name=self.source_name,
                    retrieved_at=retrieved_at,
                    raw_metadata={
                        k: v for k, v in item.items()
                        if k in (
                            "version",
                            "inputTokenLimit",
                            "outputTokenLimit",
                            "supportedGenerationMethods",
                        )
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
