"""Adapter factory — imports all adapters (triggers @register_adapter) and
provides get_adapter() as the single dispatch point for the rest of the system.

Usage:
    from tidus.adapters.adapter_factory import get_adapter

    adapter = get_adapter("anthropic")
    response = await adapter.complete(model_id, task)
"""

from __future__ import annotations

import tidus.adapters.anthropic_adapter  # noqa: F401
import tidus.adapters.deepseek_adapter  # noqa: F401
import tidus.adapters.google_adapter  # noqa: F401
import tidus.adapters.mistral_adapter  # noqa: F401
import tidus.adapters.moonshot_adapter  # noqa: F401

# Import all adapters to trigger @register_adapter side effects.
# Order: local first (no API key needed) → cloud adapters.
import tidus.adapters.ollama_adapter  # noqa: F401
import tidus.adapters.openai_adapter  # noqa: F401

# Universal OpenRouter execution adapter — serves vendors without a native
# adapter (dispatched when ModelSpec.route_id is set). Imported last.
import tidus.adapters.openrouter_adapter  # noqa: F401
import tidus.adapters.xai_adapter  # noqa: F401
from tidus.adapters.base import AbstractModelAdapter, get_adapter, list_adapters  # noqa: F401

__all__ = ["get_adapter", "list_adapters", "resolve_adapter"]


def resolve_adapter(spec) -> tuple[AbstractModelAdapter, str]:
    """Pick the (adapter, execution-model-id) for a ModelSpec.

    OpenRouter-served models (``route_id`` set) use the universal OpenRouter
    adapter with their ``route_id``. Everything else uses its native vendor
    adapter with ``model_id`` — preferred, since the direct vendor API avoids
    OpenRouter's markup. Raises ``KeyError`` when no adapter can serve the model
    (preserves the existing 501 path in ``/complete``).
    """
    route_id = getattr(spec, "route_id", None)
    if route_id:
        return get_adapter("openrouter"), route_id
    return get_adapter(spec.vendor), spec.model_id
