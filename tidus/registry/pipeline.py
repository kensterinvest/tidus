"""RegistryPipeline — orchestrates the multi-source price sync and revision promotion.

run_price_sync_cycle():
  1.  Acquire PostgreSQL advisory lock (no-op on SQLite)
  2.  Clean up stale PENDING revisions (>1h old)
  3.  Ingest: fetch from all available sources concurrently
  4.  Write PricingIngestionRunORM rows per source
  5.  Consensus: MAD outlier detection → winning quote per model
  6.  Normalize: compare against active revision; collect changed models
  7.  No changes → release lock, return None
  8.  Tier 1 + Tier 2 validation → fail → FAILED revision, return None
  9.  Phase A: insert revision (PENDING) + all entries (safe to read; router ignores PENDING)
  10. Tier 3 canary: retry logic; fail → FAILED revision, return None
  11. Phase B: atomic flip (ACTIVE→SUPERSEDED, PENDING→ACTIVE in one transaction)
  12. Write PriceChangeRecord rows (backward compat)
  13. Write audit entry
  14. Trigger EffectiveRegistry.refresh()
  15. Release lock, return PipelineResult

force_activate():
  - Tier 1 + Tier 2 only (skips Tier 3)
  - Phase B atomic flip
  - Audit entry
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import delete, select, text, update

from tidus.db.registry_orm import (
    ModelCatalogEntryORM,
    ModelCatalogRevisionORM,
    PricingIngestionRunORM,
)
from tidus.db.repositories.registry_repo import (
    get_active_revision,
    get_entries_for_revision,
    get_revision_by_id,
)
from tidus.models.model_registry import ModelSpec
from tidus.registry.validators import CanaryProbe, InvariantValidator, SchemaValidator
from tidus.sync.pricing.base import PriceQuote, PricingSource
from tidus.sync.pricing.consensus import ConsensusError, PriceConsensus

log = structlog.get_logger(__name__)

# PostgreSQL advisory lock key for the price sync job.
# Must be stable across deployments and unique to this job.
_SYNC_LOCK_KEY = 1_234_567_891

_STALE_PENDING_THRESHOLD = timedelta(hours=1)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Returned by run_price_sync_cycle() when a revision was successfully created."""

    revision_id: str
    changes: list[dict]               # [{model_id, field, old_value, new_value, delta_pct}]
    sources_used: list[str]
    single_source_models: list[str]
    ingestion_run_ids: list[str]


@dataclass
class DryRunResult:
    """Returned by run_price_sync_cycle(dry_run=True). No DB writes performed."""

    would_change: list[dict]           # same shape as PipelineResult.changes
    validation_errors: list[str]       # Tier 1 + Tier 2 error strings


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RegistryPipeline:
    """Orchestrates multi-source price sync → revision creation → promotion."""

    def __init__(self, session_factory, registry=None) -> None:
        self._sf = session_factory
        self._registry = registry  # EffectiveRegistry; used for refresh() + advisor

    # ── Main entry point ─────────────────────────────────────────────────────

    async def run_price_sync_cycle(
        self,
        sources: list[PricingSource],
        policies_path: str = "config/policies.yaml",
        dry_run: bool = False,
    ) -> PipelineResult | DryRunResult | None:
        """Run a full price sync cycle.

        Returns:
            PipelineResult  — when a new revision was successfully activated.
            DryRunResult    — when dry_run=True (no DB writes).
            None            — no changes, validation failed, or canary failed.
        """
        import os

        from tidus.utils.yaml_loader import load_yaml
        raw = load_yaml(policies_path)
        threshold = raw.get("pricing_sync", {}).get("change_threshold", 0.05)
        canary_cfg = raw.get("canary", {})
        # Allow TIDUS_CANARY_SAMPLE_SIZE=0 in dev to skip Tier 3 without editing policies.yaml
        env_sample = os.environ.get("TIDUS_CANARY_SAMPLE_SIZE")
        if env_sample is not None:
            canary_cfg = {**canary_cfg, "sample_size": int(env_sample)}

        lock_session = None
        lock_held = False

        try:
            # ── Step 1: acquire advisory lock ─────────────────────────────────
            lock_session, lock_held = await self._acquire_lock()
            if not lock_held:
                log.info("price_sync_skipped_lock_held")
                return None

            # ── Step 2: clean up stale PENDING revisions ──────────────────────
            await self._cleanup_stale_pending()

            # ── Step 3: ingest from all available sources ─────────────────────
            available = [s for s in sources if s.is_available]
            if not available:
                log.warning("price_sync_no_sources_available")
                return None

            fetch_results = await asyncio.gather(
                *[self._fetch_source(s) for s in available],
                return_exceptions=False,
            )
            # fetch_results is a list of (source, quotes, run_id, error)
            all_quotes: list[PriceQuote] = []
            ingestion_run_ids: list[str] = []
            sources_used: list[str] = []

            for source, quotes, run_id, error in fetch_results:
                ingestion_run_ids.append(run_id)
                if error is None:
                    all_quotes.extend(quotes)
                    sources_used.append(source.source_name)

            # ── Step 4: consensus ─────────────────────────────────────────────
            try:
                consensus = PriceConsensus().resolve(all_quotes)
            except ConsensusError as exc:
                log.error("price_sync_consensus_failed", error=str(exc))
                return None

            single_source_models = consensus.single_source_models

            # ── Step 5: normalize — find changed models ───────────────────────
            active_rev = await get_active_revision(self._sf)
            if active_rev is None:
                log.warning("price_sync_no_active_revision", reason="cannot compute diff")
                return None

            current_entries = await get_entries_for_revision(self._sf, active_rev.revision_id)
            current_by_id: dict[str, ModelSpec] = {}
            for entry in current_entries:
                try:
                    current_by_id[entry.model_id] = ModelSpec.model_validate(entry.spec_json)
                except Exception:
                    continue

            new_specs: dict[str, ModelSpec] = dict(current_by_id)  # start from current
            changes: list[dict] = []
            now = datetime.now(UTC)

            for model_id, spec in current_by_id.items():
                if spec.is_local:
                    continue
                quote = consensus.quotes.get(model_id)
                if quote is None:
                    continue

                for field_name, new_val, old_val in [
                    ("input_price", quote.input_price, spec.input_price),
                    ("output_price", quote.output_price, spec.output_price),
                    ("cache_read_price", quote.cache_read_price, spec.cache_read_price),
                    ("cache_write_price", quote.cache_write_price, spec.cache_write_price),
                ]:
                    if old_val == 0 and new_val == 0:
                        continue
                    ref = old_val if old_val != 0 else new_val
                    delta = (new_val - old_val) / ref   # signed: negative = price drop
                    if abs(delta) >= threshold:
                        changes.append({
                            "model_id": model_id,
                            "field": field_name,
                            "old_value": old_val,
                            "new_value": new_val,
                            "delta_pct": round(delta * 100, 2),   # signed %
                            "detected_at": now,
                        })

            # Apply all price updates to build the new spec set
            changed_model_ids = {c["model_id"] for c in changes}
            for model_id in changed_model_ids:
                spec = current_by_id[model_id]
                quote = consensus.quotes[model_id]
                new_specs[model_id] = spec.model_copy(update={
                    "input_price": quote.input_price,
                    "output_price": quote.output_price,
                    "cache_read_price": quote.cache_read_price,
                    "cache_write_price": quote.cache_write_price,
                    "last_price_check": now.date(),
                })

            # Also pick up new models: in models.yaml + price source, but not yet in DB.
            from tidus.router.registry import ModelRegistry
            from tidus.settings import get_settings
            yaml_registry = ModelRegistry.load(get_settings().models_config_path)
            yaml_by_id = {s.model_id: s for s in yaml_registry.list_all()}
            for model_id, quote in consensus.quotes.items():
                if model_id in current_by_id or model_id not in yaml_by_id:
                    continue
                yaml_spec = yaml_by_id[model_id]
                new_specs[model_id] = yaml_spec.model_copy(update={
                    "input_price": quote.input_price,
                    "output_price": quote.output_price,
                    "cache_read_price": quote.cache_read_price,
                    "cache_write_price": quote.cache_write_price,
                    "last_price_check": now.date(),
                })
                changes.append({
                    "model_id": model_id,
                    "field": "new_model",
                    "old_value": 0.0,
                    "new_value": quote.input_price,
                    "delta_pct": 100.0,
                    "detected_at": now,
                })
                log.info("price_sync_new_model_detected", model_id=model_id)

            if not changes:
                log.info("price_sync_no_changes")
                return None

            all_new_specs = list(new_specs.values())

            # ── Step 6: Tier 1 + Tier 2 validation ───────────────────────────
            spec_dicts = [s.model_dump() for s in all_new_specs]
            tier1_errors = SchemaValidator().validate(spec_dicts)
            tier2_errors = InvariantValidator().validate(all_new_specs)
            validation_errors = tier1_errors + tier2_errors

            if dry_run:
                return DryRunResult(
                    would_change=changes,
                    validation_errors=validation_errors,
                )

            if validation_errors:
                log.error("price_sync_validation_failed", errors=validation_errors)
                return None

            # ── Step 7: Phase A write — insert PENDING revision + entries ─────
            revision_id = str(uuid.uuid4())
            signature_hash = self._compute_signature(spec_dicts)
            await self._phase_a_write(revision_id, all_new_specs, signature_hash)

            # ── Step 8: Tier 3 canary ─────────────────────────────────────────
            probe = CanaryProbe(
                sample_size=canary_cfg.get("sample_size", 3),
                max_attempts=canary_cfg.get("max_attempts", 3),
                retry_delay_seconds=canary_cfg.get("retry_delay_seconds", 30.0),
                pass_rate=canary_cfg.get("pass_rate", 0.67),
            )
            passes, canary_results = await probe.run(all_new_specs)
            canary_json = [
                {
                    "model_id": r.model_id,
                    "attempts": r.attempts,
                    "successes": r.successes,
                    "failure_reasons": r.failure_reasons,
                    "verdict": r.verdict,
                }
                for r in canary_results
            ]

            if not passes:
                await self._mark_failed(revision_id, "Tier 3 canary probe failed", canary_json)
                log.error("price_sync_canary_failed", revision_id=revision_id)
                return None

            # ── Step 9: Phase B — atomic flip ─────────────────────────────────
            await self._phase_b_flip(revision_id, canary_json)

            # ── Step 10: write PriceChangeRecord rows (backward compat) ───────
            await self._write_price_change_records(changes, all_new_specs)

            # ── Step 10b: write price_market_history rows ─────────────────────
            await self._write_market_history(changes, all_new_specs, revision_id, sources_used)

            # ── Step 11: update ingestion runs with created revision_id ────────
            await self._link_ingestion_runs(ingestion_run_ids, revision_id)

            # ── Step 12: refresh in-memory registry ───────────────────────────
            if self._registry is not None and hasattr(self._registry, "refresh"):
                try:
                    await self._registry.refresh(self._sf)
                except Exception as exc:
                    log.error("price_sync_refresh_failed", error=str(exc))

            # ── Step 13: update Prometheus metrics ────────────────────────────
            try:
                from tidus.observability.metrics_updater import MetricsUpdater
                await MetricsUpdater().update(self._registry, self._sf)
            except Exception as exc:
                log.error("price_sync_metrics_update_failed", error=str(exc))

            log.info(
                "price_sync_complete",
                revision_id=revision_id,
                changes=len(changes),
                sources=sources_used,
            )
            return PipelineResult(
                revision_id=revision_id,
                changes=changes,
                sources_used=sources_used,
                single_source_models=single_source_models,
                ingestion_run_ids=ingestion_run_ids,
            )

        finally:
            if lock_held and lock_session is not None:
                await self._release_lock(lock_session)
                await lock_session.close()

    # ── force_activate ────────────────────────────────────────────────────────

    async def force_activate(
        self,
        revision_id: str,
        actor,
        justification: str,
    ) -> None:
        """Force-promote a SUPERSEDED or PENDING revision (bypasses Tier 3).

        Runs Tier 1 + Tier 2 validation. Raises ValueError on validation failure.
        Raises ValueError if the revision is not in a promotable state.
        """
        rev = await get_revision_by_id(self._sf, revision_id)
        if rev is None:
            raise ValueError(f"Revision {revision_id!r} not found")
        if rev.status not in ("superseded", "pending"):
            raise ValueError(f"Cannot force-activate a revision with status={rev.status!r}")

        entries = await get_entries_for_revision(self._sf, revision_id)
        specs = []
        spec_dicts = []
        for entry in entries:
            try:
                spec = ModelSpec.model_validate(entry.spec_json)
                specs.append(spec)
                spec_dicts.append(entry.spec_json)
            except Exception as exc:
                raise ValueError(f"Entry for {entry.model_id} failed schema validation: {exc}") from exc

        tier1_errors = SchemaValidator().validate(spec_dicts)
        tier2_errors = InvariantValidator().validate(specs)
        all_errors = tier1_errors + tier2_errors
        if all_errors:
            raise ValueError("Validation failed:\n" + "\n".join(all_errors))

        canary_json: list[dict] = []  # Tier 3 skipped
        await self._phase_b_flip(revision_id, canary_json)

        log.info(
            "force_activate_complete",
            revision_id=revision_id,
            actor=actor.sub,
            justification=justification,
        )

        if self._registry is not None and hasattr(self._registry, "refresh"):
            try:
                await self._registry.refresh(self._sf)
            except Exception as exc:
                log.error("force_activate_refresh_failed", error=str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _fetch_source(
        self, source: PricingSource
    ) -> tuple[PricingSource, list[PriceQuote], str, Exception | None]:
        """Fetch from one source, write an ingestion run row, return (source, quotes, run_id, error)."""
        run_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)
        quotes: list[PriceQuote] = []
        error: Exception | None = None
        status = "success"

        try:
            quotes = await source.fetch_quotes()
        except Exception as exc:
            error = exc
            status = "failed"
            log.error("pricing_source_fetch_failed", source=source.source_name, error=str(exc))

        completed_at = datetime.now(UTC)

        async with self._sf() as session:
            run_orm = PricingIngestionRunORM(
                run_id=run_id,
                started_at=started_at,
                completed_at=completed_at,
                source_name=source.source_name,
                status=status,
                raw_payload=None,  # don't store raw response for privacy
                model_count=len(quotes),
                quotes_valid=len(quotes),
                quotes_rejected=0,
                error_message=str(error) if error else None,
            )
            session.add(run_orm)
            await session.commit()

        return source, quotes, run_id, error

    async def _cleanup_stale_pending(self) -> None:
        """Mark PENDING revisions older than 1h as FAILED and delete their entries."""
        cutoff = datetime.now(UTC) - _STALE_PENDING_THRESHOLD
        async with self._sf() as session:
            result = await session.execute(
                select(ModelCatalogRevisionORM).where(
                    ModelCatalogRevisionORM.status == "pending",
                    ModelCatalogRevisionORM.created_at < cutoff,
                )
            )
            stale = result.scalars().all()
            if stale:
                stale_ids = [r.revision_id for r in stale]
                await session.execute(
                    delete(ModelCatalogEntryORM).where(
                        ModelCatalogEntryORM.revision_id.in_(stale_ids)
                    )
                )
                await session.execute(
                    update(ModelCatalogRevisionORM)
                    .where(ModelCatalogRevisionORM.revision_id.in_(stale_ids))
                    .values(status="failed", failure_reason="stale_pending_cleanup")
                )
                await session.commit()
                log.info("cleanup_stale_pending", count=len(stale_ids))

    async def _phase_a_write(
        self, revision_id: str, specs: list[ModelSpec], signature_hash: str
    ) -> None:
        """Phase A: insert PENDING revision + all model entries. Safe to read."""
        async with self._sf() as session:
            rev_orm = ModelCatalogRevisionORM(
                revision_id=revision_id,
                source="price_sync",
                signature_hash=signature_hash,
                status="pending",
            )
            session.add(rev_orm)

            for spec in specs:
                # mode="json" ensures date/datetime fields are serialized to strings,
                # which SQLite's JSON column type requires.
                spec_dict = spec.model_dump(mode="json")
                spec_dict["schema_version"] = 1
                entry = ModelCatalogEntryORM(
                    id=str(uuid.uuid4()),
                    revision_id=revision_id,
                    model_id=spec.model_id,
                    spec_json=spec_dict,
                    schema_version=1,
                )
                session.add(entry)

            await session.commit()
        log.debug("pipeline_phase_a_written", revision_id=revision_id, model_count=len(specs))

    async def _phase_b_flip(self, revision_id: str, canary_json: list[dict]) -> None:
        """Phase B: atomic ACTIVE→SUPERSEDED + PENDING/SUPERSEDED→ACTIVE."""
        now = datetime.now(UTC)
        async with self._sf() as session:
            await session.execute(
                update(ModelCatalogRevisionORM)
                .where(ModelCatalogRevisionORM.status == "active")
                .values(status="superseded")
            )
            await session.execute(
                update(ModelCatalogRevisionORM)
                .where(ModelCatalogRevisionORM.revision_id == revision_id)
                .values(status="active", activated_at=now, canary_results=canary_json or None)
            )
            await session.commit()
        log.info("pipeline_phase_b_flipped", revision_id=revision_id)

    async def _mark_failed(
        self, revision_id: str, reason: str, canary_json: list[dict] | None = None
    ) -> None:
        async with self._sf() as session:
            await session.execute(
                update(ModelCatalogRevisionORM)
                .where(ModelCatalogRevisionORM.revision_id == revision_id)
                .values(status="failed", failure_reason=reason, canary_results=canary_json)
            )
            await session.commit()

    async def _write_price_change_records(
        self, changes: list[dict], specs: list[ModelSpec]
    ) -> None:
        """Write PriceChangeRecord rows for backward compat with the audit dashboard."""
        spec_by_id = {s.model_id: s for s in specs}
        try:
            from tidus.db.engine import PriceChangeLogORM

            async with self._sf() as session:
                for change in changes:
                    spec = spec_by_id.get(change["model_id"])
                    vendor = spec.vendor if spec else "unknown"
                    session.add(PriceChangeLogORM(
                        id=str(uuid.uuid4()),
                        model_id=change["model_id"],
                        vendor=vendor,
                        field_changed=change["field"],
                        old_value=change["old_value"],
                        new_value=change["new_value"],
                        delta_pct=change["delta_pct"] / 100,
                        detected_at=change["detected_at"],
                        source="price_sync",
                    ))
                await session.commit()
        except Exception as exc:
            log.error("price_change_record_write_failed", error=str(exc))

    async def _write_market_history(
        self,
        changes: list[dict],
        specs: list[ModelSpec],
        revision_id: str,
        sources_used: list[str],
    ) -> None:
        """Write price_market_history rows — permanent market intelligence log."""
        from tidus.db.registry_orm import PriceMarketHistoryORM
        spec_by_id = {s.model_id: s for s in specs}
        source_label = "+".join(sources_used) if sources_used else "price_sync"
        today = datetime.now(UTC).date()
        try:
            async with self._sf() as session:
                for change in changes:
                    spec = spec_by_id.get(change["model_id"])
                    vendor = spec.vendor if spec else "unknown"
                    session.add(PriceMarketHistoryORM(
                        id=str(uuid.uuid4()),
                        model_id=change["model_id"],
                        vendor=vendor,
                        event_date=today,
                        field=change["field"],
                        old_value_usd_1m=round(change["old_value"] * 1000, 6),
                        new_value_usd_1m=round(change["new_value"] * 1000, 6),
                        delta_pct=change["delta_pct"],
                        source=source_label,
                        revision_id=revision_id,
                        reason=None,  # populated later by report generator or operator
                    ))
                await session.commit()
            log.info("market_history_written", count=len(changes), revision_id=revision_id)
        except Exception as exc:
            log.error("market_history_write_failed", error=str(exc))

    async def _link_ingestion_runs(self, run_ids: list[str], revision_id: str) -> None:
        """Update ingestion run rows with the created revision_id."""
        try:
            async with self._sf() as session:
                await session.execute(
                    update(PricingIngestionRunORM)
                    .where(PricingIngestionRunORM.run_id.in_(run_ids))
                    .values(revision_id_created=revision_id)
                )
                await session.commit()
        except Exception as exc:
            log.error("ingestion_run_link_failed", error=str(exc))

    async def write_weekly_snapshot(self, revision_id: str) -> int:
        """Write one price snapshot row per non-local model for time-series graphs.

        Called every Sunday regardless of whether prices changed. Uses
        INSERT OR IGNORE (via UniqueConstraint) so re-runs are idempotent.
        Returns the number of rows written.
        """
        from tidus.db.registry_orm import ModelPriceSnapshotORM

        entries = await get_entries_for_revision(self._sf, revision_id)
        today = datetime.now(UTC).date()
        count = 0
        try:
            async with self._sf() as session:
                for entry in entries:
                    try:
                        spec = ModelSpec.model_validate(entry.spec_json)
                    except Exception:
                        continue
                    if spec.is_local:
                        continue
                    session.add(ModelPriceSnapshotORM(
                        id=str(uuid.uuid4()),
                        snapshot_date=today,
                        model_id=spec.model_id,
                        vendor=spec.vendor,
                        input_usd_1m=round(spec.input_price * 1000, 6),
                        output_usd_1m=round(spec.output_price * 1000, 6),
                        cache_read_usd_1m=round(spec.cache_read_price * 1000, 6),
                        cache_write_usd_1m=round(spec.cache_write_price * 1000, 6),
                        revision_id=revision_id,
                    ))
                    count += 1
                await session.commit()
        except Exception as exc:
            # UniqueConstraint violation means we already snapshotted today — not an error
            if "UNIQUE" in str(exc).upper():
                log.info("weekly_snapshot_already_exists", date=today)
            else:
                log.error("weekly_snapshot_failed", error=str(exc))
            return 0
        log.info("weekly_snapshot_written", count=count, revision_id=revision_id, date=today)
        return count

    # ── Advisory lock (PostgreSQL only) ──────────────────────────────────────

    async def _acquire_lock(self):
        """Try to acquire pg_try_advisory_lock. Returns (session, acquired)."""
        from tidus.settings import get_settings
        settings = get_settings()
        if "sqlite" in settings.database_url:
            return None, True  # SQLite: no advisory locks needed

        session = self._sf()
        conn = await session.__aenter__()
        try:
            result = await conn.execute(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": _SYNC_LOCK_KEY}
            )
            acquired = result.scalar()
            return session, bool(acquired)
        except Exception as exc:
            log.warning("advisory_lock_acquire_failed", error=str(exc))
            await session.__aexit__(None, None, None)
            return None, True  # fail open: let the job proceed

    async def _release_lock(self, session) -> None:
        """Release the advisory lock on the SAME session that acquired it.

        PostgreSQL session-level advisory locks are connection-scoped: unlocking
        on a different connection is a no-op (or raises an error) and leaves the
        original lock held until the connection is recycled.
        """
        from tidus.settings import get_settings
        settings = get_settings()
        if "sqlite" in settings.database_url:
            return
        try:
            await session.execute(
                text("SELECT pg_advisory_unlock(:k)"), {"k": _SYNC_LOCK_KEY}
            )
            await session.commit()
        except Exception as exc:
            log.warning("advisory_lock_release_failed", error=str(exc))

    @staticmethod
    def _compute_signature(spec_dicts: list[dict]) -> str:
        """Compute a full SHA-256 content fingerprint of the spec set.

        Returns the complete 64-hex-char (256-bit) digest.  The previous
        16-char (64-bit) truncation fell below the birthday-attack threshold
        and was misleadingly named; the full digest is stored as-is.
        """
        import hashlib
        import json
        canonical = json.dumps(spec_dicts, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()
