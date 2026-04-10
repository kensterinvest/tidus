"""EffectiveRegistry — drop-in replacement for ModelRegistry.

Implements the same interface (get, list_all, list_enabled, upsert, set_enabled,
update_latency) so that ModelSelector, CapabilityMatcher, and all route
endpoints continue to work unchanged.

Three-layer merge at build time:
  Layer 1: base catalog   (active revision entries from DB)
  Layer 2: overrides      (RBAC-controlled, scoped, from model_overrides table)
  Layer 3: telemetry      (health probe results, from model_telemetry table)

Cache invalidation:
  The merged specs are held in _by_id (dict, same shape as ModelRegistry).
  refresh() is called every 60 seconds by TidusScheduler. It queries two cheap
  sentinel values (active_revision_id + override_checkpoint) and only rebuilds
  the full dict if either changed — making normal polling nearly free.

Fallback:
  If no active revision exists in the DB (e.g. after a failed migration or on a
  test environment without seeding), falls back to ModelRegistry.load() from
  the YAML file. This preserves v1.0.0 behavior exactly.
"""

from __future__ import annotations

import structlog

from tidus.db.repositories.registry_repo import (
    get_active_overrides,
    get_active_revision,
    get_entries_for_revision,
    get_override_checkpoint,
)
from tidus.models.model_registry import ModelSpec
from tidus.models.registry_models import TelemetrySnapshot
from tidus.registry.merge import merge_spec
from tidus.registry.telemetry_reader import TelemetryReader
from tidus.router.registry import ModelRegistry

log = structlog.get_logger(__name__)


class EffectiveRegistry:
    """Versioned, override-aware, telemetry-enhanced model registry."""

    def __init__(
        self,
        by_id: dict[str, ModelSpec],
        active_revision_id: str,
        override_checkpoint: str,
        fallback_yaml_path: str,
    ) -> None:
        self._by_id = by_id
        self._active_revision_id = active_revision_id
        self._override_checkpoint = override_checkpoint
        self._fallback_yaml_path = fallback_yaml_path

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    async def build(
        cls,
        session_factory,
        fallback_yaml_path: str = "config/models.yaml",
    ) -> EffectiveRegistry:
        """Build an EffectiveRegistry from the active DB revision.

        Falls back to YAML load if no active revision is present.
        """
        revision = await get_active_revision(session_factory)

        if revision is None:
            log.warning("effective_registry_fallback", reason="no_active_revision")
            plain = ModelRegistry.load(fallback_yaml_path)
            by_id = {s.model_id: s for s in plain.list_all()}
            return cls(
                by_id=by_id,
                active_revision_id="",
                override_checkpoint="",
                fallback_yaml_path=fallback_yaml_path,
            )

        entries = await get_entries_for_revision(session_factory, revision.revision_id)
        overrides = await get_active_overrides(session_factory)
        telemetry_map: dict[str, TelemetrySnapshot] = await TelemetryReader().get_all_snapshots(session_factory)
        checkpoint = await get_override_checkpoint(session_factory)

        by_id: dict[str, ModelSpec] = {}
        for entry in entries:
            try:
                base = ModelSpec.model_validate(entry.spec_json)
            except Exception as exc:
                log.error(
                    "effective_registry_entry_invalid",
                    model_id=entry.model_id,
                    revision_id=revision.revision_id,
                    error=str(exc),
                )
                continue

            telemetry = telemetry_map.get(base.model_id)
            merged = merge_spec(base, overrides, telemetry)
            by_id[merged.model_id] = merged

        log.info(
            "effective_registry_built",
            revision_id=revision.revision_id,
            model_count=len(by_id),
            active_overrides=len(overrides),
        )
        return cls(
            by_id=by_id,
            active_revision_id=revision.revision_id,
            override_checkpoint=checkpoint,
            fallback_yaml_path=fallback_yaml_path,
        )

    # ── Refresh (called by scheduler every 60s) ───────────────────────────────

    async def refresh(self, session_factory) -> bool:
        """Rebuild the in-memory registry if the active revision or overrides changed.

        Returns True if a rebuild occurred, False if the cache was still valid.
        The check is cheap: two small indexed queries (revision status, override count+ts).
        """
        revision = await get_active_revision(session_factory)
        new_revision_id = revision.revision_id if revision else ""
        new_checkpoint = await get_override_checkpoint(session_factory)

        if new_revision_id == self._active_revision_id and new_checkpoint == self._override_checkpoint:
            return False

        rebuilt = await EffectiveRegistry.build(session_factory, self._fallback_yaml_path)
        self._by_id = rebuilt._by_id
        self._active_revision_id = rebuilt._active_revision_id
        self._override_checkpoint = rebuilt._override_checkpoint

        log.info(
            "effective_registry_refreshed",
            new_revision_id=self._active_revision_id,
        )
        return True

    # ── ModelRegistry interface (drop-in compatibility) ───────────────────────

    @property
    def active_revision_id(self) -> str:
        return self._active_revision_id

    def get(self, model_id: str) -> ModelSpec | None:
        return self._by_id.get(model_id)

    def list_all(self) -> list[ModelSpec]:
        return list(self._by_id.values())

    def list_enabled(self) -> list[ModelSpec]:
        """Return models available for routing: enabled AND not retired.

        Deprecated models are intentionally included — the plan specifies they
        are still routed (with a score penalty applied by ModelSelector) and
        still health-probed and drift-detected during the deprecation window.
        Only hard-disabled (enabled=False) models are excluded.
        """
        return [s for s in self._by_id.values() if s.enabled]

    def upsert(self, spec: ModelSpec) -> None:
        """In-memory upsert for health probe compatibility.

        Note: these changes are ephemeral and lost on the next refresh().
        Phase 4 will write telemetry to the DB so probes persist across restarts.
        """
        self._by_id[spec.model_id] = spec

    def set_enabled(self, model_id: str, enabled: bool) -> bool:
        spec = self._by_id.get(model_id)
        if spec is None:
            return False
        self._by_id[model_id] = spec.model_copy(update={"enabled": enabled})
        return True

    def update_latency(self, model_id: str, latency_p50_ms: int) -> bool:
        spec = self._by_id.get(model_id)
        if spec is None:
            return False
        self._by_id[model_id] = spec.model_copy(update={"latency_p50_ms": latency_p50_ms})
        return True

    def __len__(self) -> int:
        return len(self._by_id)

    def __repr__(self) -> str:
        enabled = sum(1 for s in self._by_id.values() if s.enabled)
        return (
            f"EffectiveRegistry(revision={self._active_revision_id!r}, "
            f"total={len(self._by_id)}, enabled={enabled})"
        )
