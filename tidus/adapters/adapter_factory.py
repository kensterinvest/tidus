"""Adapter factory — imports all adapters (triggers @register_adapter) and
provides get_adapter() as the single dispatch point for the rest of the system.

Usage:
    from tidus.adapters.adapter_factory import get_adapter

    adapter = get_adapter("anthropic")
    response = await adapter.complete(model_id, task)
"""

from __future__ import annotations

# Import all adapters to trigger @register_adapter side effects.
# Order: local first (no API key needed) → cloud adapters.
import tidus.adapters.ollama_adapter      # noqa: F401
import tidus.adapters.anthropic_adapter   # noqa: F401
import tidus.adapters.openai_adapter      # noqa: F401
import tidus.adapters.google_adapter      # noqa: F401
import tidus.adapters.mistral_adapter     # noqa: F401
import tidus.adapters.deepseek_adapter    # noqa: F401
import tidus.adapters.xai_adapter         # noqa: F401
import tidus.adapters.moonshot_adapter    # noqa: F401

from tidus.adapters.base import get_adapter, list_adapters  # noqa: F401

__all__ = ["get_adapter", "list_adapters"]
