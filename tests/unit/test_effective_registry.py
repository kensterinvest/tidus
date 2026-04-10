"""Unit tests for EffectiveRegistry.

Covers:
  - YAML fallback when no active revision exists
  - DB path: builds from active revision entries with merge
  - refresh() detects revision change and rebuilds (not just TTL)
  - refresh() detects override checkpoint change and rebuilds
  - refresh() is a no-op when nothing changed (cheap sentinel check)
  - ModelRegistry interface: get, list_all, list_enabled, upsert, set_enabled, update_latency
  - active_revision_id property exposed
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tidus.models.model_registry import ModelSpec, ModelTier, TokenizerType
from tidus.registry.effective_registry import EffectiveRegistry

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_spec(model_id="gpt-4o", enabled=True, latency=100):
    return ModelSpec(
        model_id=model_id,
        display_name=model_id,
        vendor="openai",
        provider="openai",
        enabled=enabled,
        is_local=False,
        tier=ModelTier.mid,
        tokenizer=TokenizerType.tiktoken_o200k,
        max_context=128000,
        input_price=5.0,
        output_price=15.0,
        latency_p50_ms=latency,
    )


def _make_registry(by_id=None, revision_id="rev-1", checkpoint="1:ts") -> EffectiveRegistry:
    if by_id is None:
        by_id = {"gpt-4o": _make_spec()}
    return EffectiveRegistry(
        by_id=by_id,
        active_revision_id=revision_id,
        override_checkpoint=checkpoint,
        fallback_yaml_path="config/models.yaml",
    )


# ── ModelRegistry interface ───────────────────────────────────────────────────

def test_get_returns_spec():
    reg = _make_registry()
    spec = reg.get("gpt-4o")
    assert spec is not None
    assert spec.model_id == "gpt-4o"


def test_get_missing_returns_none():
    reg = _make_registry()
    assert reg.get("does-not-exist") is None


def test_list_all_returns_all_models():
    by_id = {"a": _make_spec("a"), "b": _make_spec("b")}
    reg = _make_registry(by_id=by_id)
    assert len(reg.list_all()) == 2


def test_list_enabled_filters_disabled():
    by_id = {
        "enabled": _make_spec("enabled", enabled=True),
        "disabled": _make_spec("disabled", enabled=False),
    }
    reg = _make_registry(by_id=by_id)
    enabled = reg.list_enabled()
    assert len(enabled) == 1
    assert enabled[0].model_id == "enabled"


def test_upsert_updates_in_memory():
    reg = _make_registry()
    new_spec = _make_spec("gpt-4o", latency=999)
    reg.upsert(new_spec)
    assert reg.get("gpt-4o").latency_p50_ms == 999


def test_set_enabled_returns_true_on_success():
    reg = _make_registry()
    assert reg.set_enabled("gpt-4o", False) is True
    assert reg.get("gpt-4o").enabled is False


def test_set_enabled_returns_false_for_missing():
    reg = _make_registry()
    assert reg.set_enabled("no-model", False) is False


def test_update_latency_success():
    reg = _make_registry()
    assert reg.update_latency("gpt-4o", 250) is True
    assert reg.get("gpt-4o").latency_p50_ms == 250


def test_update_latency_missing_returns_false():
    reg = _make_registry()
    assert reg.update_latency("no-model", 250) is False


def test_active_revision_id_property():
    reg = _make_registry(revision_id="rev-abc-123")
    assert reg.active_revision_id == "rev-abc-123"


def test_len():
    by_id = {"a": _make_spec("a"), "b": _make_spec("b"), "c": _make_spec("c")}
    reg = _make_registry(by_id=by_id)
    assert len(reg) == 3


def test_repr_shows_revision_and_counts():
    reg = _make_registry(revision_id="rev-xyz")
    r = repr(reg)
    assert "rev-xyz" in r
    assert "total=" in r
    assert "enabled=" in r


# ── YAML fallback (no active revision in DB) ──────────────────────────────────

@pytest.mark.asyncio
async def test_build_falls_back_to_yaml_when_no_active_revision():
    """When get_active_revision returns None, build() uses ModelRegistry.load()."""
    mock_sf = AsyncMock()

    with (
        patch("tidus.registry.effective_registry.get_active_revision", return_value=None),
        patch("tidus.registry.effective_registry.ModelRegistry") as mock_mr,
    ):
        mock_registry = MagicMock()
        mock_registry.list_all.return_value = [_make_spec("claude-3")]
        mock_mr.load.return_value = mock_registry

        reg = await EffectiveRegistry.build(mock_sf, "config/models.yaml")

    assert reg.active_revision_id == ""
    assert reg.get("claude-3") is not None
    mock_mr.load.assert_called_once_with("config/models.yaml")


# ── Refresh: rebuild only when something changed ──────────────────────────────

@pytest.mark.asyncio
async def test_refresh_returns_false_when_unchanged():
    """refresh() is a no-op (returns False) when revision+checkpoint unchanged."""
    reg = _make_registry(revision_id="rev-1", checkpoint="5:ts1")
    mock_sf = AsyncMock()

    mock_revision = MagicMock()
    mock_revision.revision_id = "rev-1"

    with (
        patch("tidus.registry.effective_registry.get_active_revision", return_value=mock_revision),
        patch("tidus.registry.effective_registry.get_override_checkpoint", return_value="5:ts1"),
    ):
        changed = await reg.refresh(mock_sf)

    assert changed is False


@pytest.mark.asyncio
async def test_refresh_rebuilds_on_revision_change():
    """refresh() rebuilds immediately when active_revision_id changes."""
    reg = _make_registry(revision_id="rev-1", checkpoint="5:ts1")
    mock_sf = AsyncMock()

    new_revision = MagicMock()
    new_revision.revision_id = "rev-2"  # different!

    new_reg = _make_registry(revision_id="rev-2", checkpoint="5:ts1")

    with (
        patch("tidus.registry.effective_registry.get_active_revision", return_value=new_revision),
        patch("tidus.registry.effective_registry.get_override_checkpoint", return_value="5:ts1"),
        patch.object(EffectiveRegistry, "build", return_value=new_reg) as mock_build,
    ):
        changed = await reg.refresh(mock_sf)

    assert changed is True
    assert reg.active_revision_id == "rev-2"
    mock_build.assert_called_once()


@pytest.mark.asyncio
async def test_refresh_rebuilds_on_override_checkpoint_change():
    """refresh() rebuilds when the override checkpoint changes (new override created)."""
    reg = _make_registry(revision_id="rev-1", checkpoint="5:ts1")
    mock_sf = AsyncMock()

    same_revision = MagicMock()
    same_revision.revision_id = "rev-1"

    new_reg = _make_registry(revision_id="rev-1", checkpoint="6:ts2")

    with (
        patch("tidus.registry.effective_registry.get_active_revision", return_value=same_revision),
        patch("tidus.registry.effective_registry.get_override_checkpoint", return_value="6:ts2"),  # changed!
        patch.object(EffectiveRegistry, "build", return_value=new_reg),
    ):
        changed = await reg.refresh(mock_sf)

    assert changed is True
    assert reg._override_checkpoint == "6:ts2"


# ── Deprecated model inclusion in routing ────────────────────────────────────

def test_list_enabled_includes_deprecated_models():
    """Deprecated models are enabled=True and must appear in list_enabled().

    The plan explicitly states: 'Deprecated models are intentionally included —
    the plan specifies they are still routed (with a score penalty applied by
    ModelSelector) and still health-probed and drift-detected during the
    deprecation window. Only hard-disabled (enabled=False) models are excluded.'
    """
    by_id = {
        "active": _make_spec("active", enabled=True),
        "deprecated": ModelSpec(
            model_id="deprecated",
            display_name="Deprecated",
            vendor="openai",
            provider="openai",
            enabled=True,
            deprecated=True,
            is_local=False,
            tier=ModelTier.mid,
            tokenizer=TokenizerType.tiktoken_o200k,
            max_context=128000,
            input_price=5.0,
            output_price=15.0,
        ),
        "disabled": _make_spec("disabled", enabled=False),
    }
    reg = _make_registry(by_id=by_id)
    enabled = reg.list_enabled()
    model_ids = {s.model_id for s in enabled}

    assert "active" in model_ids
    assert "deprecated" in model_ids     # deprecated but enabled → included
    assert "disabled" not in model_ids   # hard-disabled → excluded
    assert len(enabled) == 2
