"""OpenAI-compatible discovery — covers vendors whose `/v1/models` endpoint
follows the OpenAI shape (Bearer auth, `{data: [{id, owned_by, ...}]}`).

Used for: openai, mistral, deepseek, xai, moonshot, groq, cohere.
Anthropic uses a different shape — see `anthropic.py`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import ClassVar

import httpx
import structlog

from tidus.sync.discovery.base import DiscoveredModel, DiscoverySource

log = structlog.get_logger(__name__)


def _strip_openai_prefix(model_id: str) -> str:
    """OpenAI ids match our canonical ids directly (e.g. 'gpt-4.1')."""
    return model_id


def _strip_mistral_aliases(model_id: str) -> str:
    """Mistral aliases ('-latest', date suffixes like '-2407') back to canonical."""
    out = re.sub(r"-latest$", "", model_id)
    out = re.sub(r"-\d{4}$", "", out)
    return out


_DEEPSEEK_ALIASES = {
    "deepseek-chat": "deepseek-v3",
    "deepseek-reasoner": "deepseek-r1",
}


def _canonicalize_deepseek(model_id: str) -> str:
    return _DEEPSEEK_ALIASES.get(model_id, model_id)


def _canonicalize_xai(model_id: str) -> str:
    """xAI ids: 'grok-4-0709' → 'grok-4'. Date/build suffixes stripped."""
    return re.sub(r"-\d{4}(?:-\d+)?$", "", model_id)


class OpenAICompatibleDiscoverySource(DiscoverySource):
    """One source class, configured per-vendor via constructor args.

    Vendors with auth/shape that match this template:
        GET <base_url>/v1/models
        Authorization: Bearer <api_key>
        Response: {"data": [{"id": "...", "owned_by": "...", ...}, ...]}
    """

    # Filter rules used by every vendor — exclude embeddings, transcription,
    # image, audio, fine-tuned, and deprecated/snapshot model variants.
    _UNWANTED_PATTERNS: ClassVar[tuple[re.Pattern[str], ...]] = (
        re.compile(r"embedding", re.I),
        re.compile(r"whisper", re.I),
        re.compile(r"tts|audio", re.I),
        re.compile(r"dall-?e|image|vision-only", re.I),
        re.compile(r"^ft:", re.I),  # fine-tuned snapshots
        re.compile(r"moderation", re.I),
    )

    def __init__(
        self,
        *,
        source_name: str,
        vendor: str,
        base_url: str,
        api_key: str,
        canonicalize: Callable[[str], str] | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._source_name = source_name
        self._vendor = vendor
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._canonicalize = canonicalize or (lambda mid: mid)
        self._timeout = timeout_seconds

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def vendor(self) -> str:
        return self._vendor

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    async def list_models(self) -> list[DiscoveredModel]:
        url = f"{self._base_url}/v1/models"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            log.warning(
                "discovery_fetch_failed",
                source=self._source_name,
                vendor=self._vendor,
                error=str(exc),
            )
            return []
        except ValueError as exc:  # JSON decode
            log.warning(
                "discovery_parse_failed",
                source=self._source_name,
                vendor=self._vendor,
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
            if any(p.search(raw_id) for p in self._UNWANTED_PATTERNS):
                continue
            canonical_id = self._canonicalize(raw_id)
            out.append(
                DiscoveredModel(
                    model_id=canonical_id,
                    vendor_id=raw_id,
                    vendor=self._vendor,
                    display_name=item.get("name") or item.get("display_name"),
                    source_name=self._source_name,
                    retrieved_at=retrieved_at,
                    raw_metadata={
                        k: v for k, v in item.items()
                        if k in ("created", "owned_by", "context_length", "type")
                    },
                )
            )

        log.info(
            "discovery_fetched",
            source=self._source_name,
            vendor=self._vendor,
            raw_count=len(items),
            kept_count=len(out),
        )
        return out


# ── Factory helpers — wire each vendor with the right base URL + canonicalizer ──

def openai_source(api_key: str, *, base_url: str = "https://api.openai.com") -> OpenAICompatibleDiscoverySource:
    return OpenAICompatibleDiscoverySource(
        source_name="openai-models",
        vendor="openai",
        base_url=base_url,
        api_key=api_key,
        canonicalize=_strip_openai_prefix,
    )


def mistral_source(api_key: str, *, base_url: str = "https://api.mistral.ai") -> OpenAICompatibleDiscoverySource:
    return OpenAICompatibleDiscoverySource(
        source_name="mistral-models",
        vendor="mistral",
        base_url=base_url,
        api_key=api_key,
        canonicalize=_strip_mistral_aliases,
    )


def deepseek_source(api_key: str, *, base_url: str = "https://api.deepseek.com") -> OpenAICompatibleDiscoverySource:
    return OpenAICompatibleDiscoverySource(
        source_name="deepseek-models",
        vendor="deepseek",
        base_url=base_url,
        api_key=api_key,
        canonicalize=_canonicalize_deepseek,
    )


def xai_source(api_key: str, *, base_url: str = "https://api.x.ai") -> OpenAICompatibleDiscoverySource:
    return OpenAICompatibleDiscoverySource(
        source_name="xai-models",
        vendor="xai",
        base_url=base_url,
        api_key=api_key,
        canonicalize=_canonicalize_xai,
    )
