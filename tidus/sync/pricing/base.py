"""Base abstractions for pricing sources.

PricingSource — ABC that all price feeds must implement.
PriceQuote    — A single price observation from one source for one model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class PriceQuote:
    """A single price observation returned by a PricingSource.

    All prices are in USD per 1K tokens. 0.0 is valid for local models.
    """

    model_id: str
    input_price: float
    output_price: float
    cache_read_price: float
    cache_write_price: float
    currency: str
    effective_date: date
    retrieved_at: datetime
    source_name: str
    source_confidence: float  # 0.0–1.0; used to pick winner in consensus
    evidence_url: str | None = None  # e.g. link to vendor pricing page


class PricingSource(ABC):
    """Abstract base class for all pricing data providers.

    Implementations:
      HardcodedSource       — built-in verified prices, always available
      TidusPricingFeedSource — optional remote pricing feed
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this source (used in audit trails)."""
        ...

    @property
    @abstractmethod
    def confidence(self) -> float:
        """Baseline confidence score for this source's quotes (0.0–1.0)."""
        ...

    @property
    def is_available(self) -> bool:
        """Return True if this source should be included in the current sync cycle.

        Implementations may return False when the feed is disabled (e.g., no URL
        configured) or the circuit breaker is OPEN.
        """
        return True

    @abstractmethod
    async def fetch_quotes(self) -> list[PriceQuote]:
        """Return all available price quotes. Empty list = no data (never raises)."""
        ...
