"""DiscoveryRunner — orchestrates per-vendor sources, diffs against the
active model registry, persists first-seen state, and produces a
DiscoveryReport for downstream rendering.

State file format (`reports/discovered_models.json`):
    {
        "<canonical_id>": {
            "vendor": "openai",
            "vendor_id": "gpt-4.1-mini",
            "display_name": "...",
            "first_seen": "2026-04-30T01:23:45+00:00",
            "last_seen":  "2026-04-30T01:23:45+00:00",
            "source": "openai-models",
            "in_registry": false,
            "raw_metadata": {...}
        },
        ...
    }

Surfacing rules in the report:
  * `new_this_run`: canonical ids seen for the first time on this run.
    These are the freshly-discovered candidates.
  * `pending_review`: ids previously discovered but still NOT in the
    Tidus registry — kept so a reviewer can see the backlog.
  * `removed_from_vendor`: ids that were in the state file last run
    but no vendor returned them this run. Often benign (rate-limit
    truncation), but a sustained absence suggests deprecation.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

from tidus.sync.discovery.base import DiscoveredModel, DiscoverySource

log = structlog.get_logger(__name__)


@dataclass
class DiscoveryReport:
    """Summary of a discovery run for the pricing report + email."""

    generated_at: datetime
    sources_run: list[str]
    sources_skipped: list[str]            # unavailable (no API key, etc.)
    new_this_run: list[DiscoveredModel]    # first-seen this cycle
    pending_review: list[DiscoveredModel]  # known + not yet in registry
    removed_from_vendor: list[str]         # ids absent this run (canonical_id only)
    total_discovered: int                  # total unique canonical ids known

    @property
    def has_findings(self) -> bool:
        return bool(self.new_this_run or self.removed_from_vendor)


class DiscoveryRunner:
    """Drives discovery sources and reconciles results with the registry.

    Pure orchestration — no DB writes. State lives in a JSON sidecar so
    the cron commits it alongside the pricing reports.
    """

    def __init__(
        self,
        sources: list[DiscoverySource],
        *,
        state_path: Path,
        registry_model_ids: Iterable[str],
    ) -> None:
        self._sources = sources
        self._state_path = state_path
        self._registry_ids = frozenset(registry_model_ids)

    async def run(self) -> DiscoveryReport:
        available = [s for s in self._sources if s.is_available]
        skipped = [s.source_name for s in self._sources if not s.is_available]

        # All vendors run concurrently — single slow vendor doesn't gate the rest.
        results = await asyncio.gather(
            *(s.list_models() for s in available),
            return_exceptions=True,
        )

        all_models: dict[str, DiscoveredModel] = {}
        for src, result in zip(available, results):
            if isinstance(result, BaseException):
                log.warning(
                    "discovery_source_raised",
                    source=src.source_name,
                    error=str(result),
                )
                continue
            for m in result:
                # Last write wins on duplicate canonical ids — sources are
                # ordered by configuration, so the first-listed source for
                # a vendor takes precedence.
                all_models.setdefault(m.model_id, m)

        prior_state = self._load_state()
        now = datetime.now(UTC)
        merged_state: dict[str, dict] = {}
        new_this_run: list[DiscoveredModel] = []
        pending_review: list[DiscoveredModel] = []

        for canonical_id, model in all_models.items():
            in_registry = canonical_id in self._registry_ids
            prior = prior_state.get(canonical_id)
            first_seen = prior["first_seen"] if prior else now.isoformat()

            merged_state[canonical_id] = {
                "vendor": model.vendor,
                "vendor_id": model.vendor_id,
                "display_name": model.display_name,
                "first_seen": first_seen,
                "last_seen": now.isoformat(),
                "source": model.source_name,
                "in_registry": in_registry,
                "raw_metadata": model.raw_metadata,
            }

            if prior is None:
                if not in_registry:
                    new_this_run.append(model)
            elif not in_registry:
                pending_review.append(model)

        # Anything in prior_state but missing from current vendor results
        removed = [k for k in prior_state if k not in all_models]
        # Preserve their record so we don't lose the first_seen timestamp
        # entirely — but stamp last_seen with the prior value so the gap
        # is visible.
        for canonical_id in removed:
            merged_state[canonical_id] = {
                **prior_state[canonical_id],
                "in_registry": canonical_id in self._registry_ids,
            }

        self._save_state(merged_state)

        report = DiscoveryReport(
            generated_at=now,
            sources_run=[s.source_name for s in available],
            sources_skipped=skipped,
            new_this_run=sorted(new_this_run, key=lambda m: (m.vendor, m.model_id)),
            pending_review=sorted(pending_review, key=lambda m: (m.vendor, m.model_id)),
            removed_from_vendor=sorted(removed),
            total_discovered=len(merged_state),
        )

        log.info(
            "discovery_complete",
            sources_run=report.sources_run,
            sources_skipped=report.sources_skipped,
            new_this_run=len(report.new_this_run),
            pending_review=len(report.pending_review),
            removed_from_vendor=len(report.removed_from_vendor),
            total_discovered=report.total_discovered,
        )
        return report

    # ── State persistence ──────────────────────────────────────────────

    def _load_state(self) -> dict[str, dict]:
        if not self._state_path.exists():
            return {}
        try:
            with self._state_path.open(encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                log.warning("discovery_state_invalid", path=str(self._state_path))
                return {}
            return data
        except (OSError, ValueError) as exc:
            log.warning(
                "discovery_state_load_failed",
                path=str(self._state_path),
                error=str(exc),
            )
            return {}

    def _save_state(self, state: dict[str, dict]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp then rename — never leave a half-written file
        # on disk if the process is killed mid-write.
        tmp = self._state_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        tmp.replace(self._state_path)
