"""Abstract model adapter interface.

Every vendor adapter inherits AbstractModelAdapter and registers itself via
@register_adapter so the AdapterFactory can dispatch by vendor name.

Example (defining a new adapter):
    @register_adapter
    class MyAdapter(AbstractModelAdapter):
        vendor = "myvendor"
        supported_model_ids = ["my-model-v1"]

        async def complete(self, model_id, task) -> AdapterResponse: ...
        async def stream_complete(self, model_id, task): ...
        async def health_check(self, model_id) -> bool: ...
        async def count_tokens(self, model_id, messages) -> int: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class AdapterResponse:
    """Normalised response returned by every adapter."""

    model_id: str
    content: str                       # generated text
    input_tokens: int                  # actual tokens consumed (from vendor)
    output_tokens: int
    latency_ms: float
    finish_reason: str = "stop"        # "stop" | "length" | "content_filter"
    raw: dict = field(default_factory=dict)  # raw vendor response for debugging


class AbstractModelAdapter(ABC):
    """Base class all vendor adapters must implement."""

    vendor: str                         # class-level constant, e.g. "anthropic"
    supported_model_ids: list[str] = [] # informational; not used for routing

    @abstractmethod
    async def complete(self, model_id: str, task) -> AdapterResponse:
        """Execute a non-streaming completion."""

    @abstractmethod
    async def stream_complete(
        self, model_id: str, task
    ) -> AsyncIterator[str]:
        """Execute a streaming completion, yielding text chunks."""

    @abstractmethod
    async def health_check(self, model_id: str) -> bool:
        """Return True if the model is reachable and responding."""

    @abstractmethod
    async def count_tokens(self, model_id: str, messages: list[dict]) -> int:
        """Return the token count for the given messages."""


# ── Registry ──────────────────────────────────────────────────────────────────

_ADAPTER_REGISTRY: dict[str, AbstractModelAdapter] = {}


def register_adapter(cls: type[AbstractModelAdapter]) -> type[AbstractModelAdapter]:
    """Class decorator — instantiates the adapter and registers it by vendor."""
    instance = cls()
    _ADAPTER_REGISTRY[cls.vendor] = instance
    return cls


def get_adapter(vendor: str) -> AbstractModelAdapter:
    """Return the adapter for a vendor name.

    Raises:
        KeyError: if no adapter is registered for the vendor.
    """
    if vendor not in _ADAPTER_REGISTRY:
        raise KeyError(
            f"No adapter registered for vendor {vendor!r}. "
            f"Available: {sorted(_ADAPTER_REGISTRY)}"
        )
    return _ADAPTER_REGISTRY[vendor]


def list_adapters() -> list[str]:
    """Return all registered vendor names."""
    return sorted(_ADAPTER_REGISTRY.keys())
