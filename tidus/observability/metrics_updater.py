"""MetricsUpdater — refreshes registry Gauges every 5 minutes.

Updates the 6 Gauge metrics from DB state. The 3 Counters are incremented
directly at the point of the operation (HealthProbe, DriftEngine) and do not
need periodic refresh.

Called:
  1. At app startup (initial population before the first scrape)
  2. Every 5 minutes by TidusScheduler (same interval as the health probe)
  3. After a successful price sync cycle (pipeline.py)

Staleness definition: a model's price data is "stale" if last_price_check is
more than STALE_DAYS days ago (default 8). Stale models get confidence=0.5.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select

from tidus.db.registry_orm import ModelCatalogRevisionORM, PricingIngestionRunORM
from tidus.observability.registry_metrics import (
    REGISTRY_ACTIVE_REVISION_ID,
    REGISTRY_ACTIVE_REVISION_TS,
    REGISTRY_LAST_SYNC_TS,
    REGISTRY_MODEL_CONFIDENCE,
    REGISTRY_MODEL_PRICE_UPDATE_TS,
    REGISTRY_STALE_MODEL_COUNT,
    revision_id_to_int,
)

log = structlog.get_logger(__name__)

STALE_DAYS = 8


class MetricsUpdater:
    """Updates all 6 registry Gauge metrics from DB and registry state."""

    async def update(self, registry, session_factory) -> None:
        """Refresh all Gauge metrics. Non-fatal — exceptions are logged."""
        try:
            await self._update_revision_metrics(session_factory)
        except Exception as exc:
            log.error("metrics_update_revision_failed", error=str(exc))

        try:
            await self._update_sync_timestamp(session_factory)
        except Exception as exc:
            log.error("metrics_update_sync_ts_failed", error=str(exc))

        try:
            self._update_model_metrics(registry)
        except Exception as exc:
            log.error("metrics_update_model_failed", error=str(exc))

    async def _update_revision_metrics(self, session_factory) -> None:
        """Set active revision hash and activation timestamp."""
        async with session_factory() as session:
            result = await session.execute(
                select(ModelCatalogRevisionORM)
                .where(ModelCatalogRevisionORM.status == "active")
                .limit(1)
            )
            revision = result.scalars().first()

        if revision is None:
            return

        REGISTRY_ACTIVE_REVISION_ID.set(revision_id_to_int(revision.revision_id))

        if revision.activated_at is not None:
            ts = revision.activated_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            REGISTRY_ACTIVE_REVISION_TS.set(ts.timestamp())

    async def _update_sync_timestamp(self, session_factory) -> None:
        """Set last_successful_sync_timestamp from pricing_ingestion_runs."""
        async with session_factory() as session:
            result = await session.execute(
                select(PricingIngestionRunORM.completed_at)
                .where(PricingIngestionRunORM.status == "success")
                .order_by(PricingIngestionRunORM.completed_at.desc())
                .limit(1)
            )
            completed_at = result.scalar_one_or_none()

        if completed_at is not None:
            ts = completed_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            REGISTRY_LAST_SYNC_TS.set(ts.timestamp())

    def _update_model_metrics(self, registry) -> None:
        """Set per-model price update timestamp, confidence, and stale count."""
        if registry is None:
            return

        stale_cutoff = datetime.now(UTC) - timedelta(days=STALE_DAYS)
        stale_count = 0

        for spec in registry.list_all():
            model_id = spec.model_id

            # last_price_check is a date; convert to datetime for comparison
            if spec.last_price_check is not None:
                from datetime import date as _date
                lpc = spec.last_price_check
                if isinstance(lpc, _date):
                    lpc_dt = datetime(lpc.year, lpc.month, lpc.day, tzinfo=UTC)
                else:
                    lpc_dt = lpc.replace(tzinfo=UTC) if lpc.tzinfo is None else lpc

                REGISTRY_MODEL_PRICE_UPDATE_TS.labels(model_id=model_id).set(
                    lpc_dt.timestamp()
                )

                is_stale = lpc_dt < stale_cutoff
                confidence = 0.5 if is_stale else 1.0
                if is_stale:
                    stale_count += 1
            else:
                confidence = 0.5
                stale_count += 1

            REGISTRY_MODEL_CONFIDENCE.labels(model_id=model_id).set(confidence)

        REGISTRY_STALE_MODEL_COUNT.set(stale_count)
