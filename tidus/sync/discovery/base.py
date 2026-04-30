"""Discovery primitives — DiscoverySource ABC and DiscoveredModel record."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DiscoveredModel:
    """A single model identifier scraped from a vendor's catalog endpoint.

    Pricing is intentionally not captured here — discovery is a SURFACE
    signal, not a pricing source. The maintainer reviews the model and
    adds it to `hardcoded_source.py` (with verified pricing) and
    `config/models.yaml` (with capabilities) to promote it.
    """

    model_id: str            # canonical id used inside Tidus (vendor-prefix stripped)
    vendor_id: str           # raw id as returned by the vendor API (audit trail)
    vendor: str              # canonical vendor name (e.g. "openai", "anthropic")
    display_name: str | None
    source_name: str         # the DiscoverySource that emitted this record
    retrieved_at: datetime
    raw_metadata: dict = field(default_factory=dict)


class DiscoverySource(ABC):
    """Abstract base for vendor-catalog discovery sources.

    Implementations call a vendor `/v1/models`-style endpoint and return
    a list of DiscoveredModel records. They should NEVER raise on network
    errors — return an empty list and log instead, so the weekly job can
    survive a single vendor outage.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Stable identifier for this source — appears in audit trails."""
        ...

    @property
    @abstractmethod
    def vendor(self) -> str:
        """Canonical vendor name this source covers."""
        ...

    @property
    def is_available(self) -> bool:
        """Return False to skip this source for the current cycle.

        Typical reason: required API key isn't configured. The runner
        treats unavailable sources as expected — no log noise.
        """
        return True

    @abstractmethod
    async def list_models(self) -> list[DiscoveredModel]:
        """Fetch the vendor's current model catalog.

        Returns an empty list on auth failure, network error, parse
        error, or anything else short of a programming bug — the
        runner should never propagate vendor outages.
        """
        ...
