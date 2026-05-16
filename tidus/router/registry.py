"""Model registry — loads models.yaml into typed ModelSpec objects at startup.

The registry is a singleton loaded once per process lifetime. All routing
decisions read from this in-memory registry rather than re-parsing YAML on
every request.

Example:
    registry = ModelRegistry.load("config/models.yaml")
    specs = registry.list_enabled()
    spec = registry.get("claude-haiku-4-5")
"""

from pathlib import Path
from typing import Any

from tidus.models.model_registry import ModelSpec
from tidus.utils.yaml_loader import load_yaml


class ModelRegistry:
    """In-memory model registry backed by models.yaml."""

    def __init__(self, specs: list[ModelSpec]) -> None:
        self._by_id: dict[str, ModelSpec] = {s.model_id: s for s in specs}

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        path: str | Path = "config/models.yaml",
        auto_path: str | Path | None = "config/models.auto.yaml",
    ) -> "ModelRegistry":
        """Load registry from one or two YAML files.

        The primary `path` is the hand-curated catalog — the source of truth
        for every model an operator has explicitly vetted. `auto_path`, when
        present on disk, is an auto-generated catalog produced by the
        weekly sync's auto-promote pass (see tidus/sync/auto_promote.py);
        it carries models surfaced by discovery that have live pricing but
        haven't been hand-reviewed yet.

        Conflict resolution: hand-curated entries always win. If a model_id
        appears in both files, the primary path's spec is kept and the auto
        spec is silently dropped. This means an operator can promote an
        auto-discovered model to "vetted" status simply by adding it to
        `models.yaml` — the auto entry then becomes a harmless duplicate
        that the next sync will rewrite or remove.

        Raises FileNotFoundError if `path` is missing (auto_path missing is
        fine — treated as empty). Raises ValueError on any validation error.
        """
        raw: dict[str, Any] = load_yaml(path)
        entries: list[dict] = raw.get("models", [])
        specs = [ModelSpec.model_validate(entry) for entry in entries]
        primary_ids = {s.model_id for s in specs}

        if auto_path is not None:
            auto_p = Path(auto_path)
            if auto_p.exists():
                auto_raw: dict[str, Any] = load_yaml(auto_p)
                for entry in auto_raw.get("models", []):
                    spec = ModelSpec.model_validate(entry)
                    if spec.model_id in primary_ids:
                        continue
                    specs.append(spec)

        return cls(specs)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get(self, model_id: str) -> ModelSpec | None:
        """Return a ModelSpec by model_id, or None if not found."""
        return self._by_id.get(model_id)

    def list_all(self) -> list[ModelSpec]:
        """Return all registered models (including disabled/deprecated)."""
        return list(self._by_id.values())

    def list_enabled(self) -> list[ModelSpec]:
        """Return all models that are enabled and not deprecated."""
        return [s for s in self._by_id.values() if s.enabled and not s.deprecated]

    def upsert(self, spec: ModelSpec) -> None:
        """Insert or replace a ModelSpec (used by price_sync and health_probe)."""
        self._by_id[spec.model_id] = spec

    def set_enabled(self, model_id: str, enabled: bool) -> bool:
        """Enable or disable a model. Returns False if model_id not found."""
        spec = self._by_id.get(model_id)
        if spec is None:
            return False
        self._by_id[model_id] = spec.model_copy(update={"enabled": enabled})
        return True

    def update_latency(self, model_id: str, latency_p50_ms: int) -> bool:
        """Update the observed median latency for a model. Returns False if not found."""
        spec = self._by_id.get(model_id)
        if spec is None:
            return False
        self._by_id[model_id] = spec.model_copy(update={"latency_p50_ms": latency_p50_ms})
        return True

    def __len__(self) -> int:
        return len(self._by_id)

    def __repr__(self) -> str:
        enabled = sum(1 for s in self._by_id.values() if s.enabled)
        return f"ModelRegistry(total={len(self._by_id)}, enabled={enabled})"
