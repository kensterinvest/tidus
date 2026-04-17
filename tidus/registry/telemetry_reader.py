"""TelemetryReader — reads the most recent health-probe measurement per model.

Implements the three-tier staleness policy:
  <24h   → "fresh"   — use latency_p50_ms in the merge layer
  24–72h → "unknown" — fall back to base catalog value; log warning
  >72h   → "expired" — excluded from merge entirely; log error

A telemetry outage (all probes stale) cannot cascade into mass model
disablement — models are NOT auto-disabled due to missing/stale telemetry.
Only active probe failures (consecutive_failures > 0 in fresh telemetry) are
used for disable decisions. Those decisions live in the drift engine (Phase 4).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select

from tidus.db.registry_orm import ModelTelemetryORM
from tidus.models.registry_models import TelemetrySnapshot

log = structlog.get_logger(__name__)

_FRESH_WINDOW = timedelta(hours=24)
_UNKNOWN_WINDOW = timedelta(hours=72)


def _classify_staleness(measured_at: datetime) -> str:
    age = datetime.now(UTC) - measured_at.replace(tzinfo=UTC) if measured_at.tzinfo is None else datetime.now(UTC) - measured_at
    if age < _FRESH_WINDOW:
        return "fresh"
    if age < _UNKNOWN_WINDOW:
        return "unknown"
    return "expired"


class TelemetryReader:
    """Reads latest telemetry per model from the DB with staleness classification."""

    async def get_all_snapshots(self, session_factory) -> dict[str, TelemetrySnapshot]:
        """Return the most recent TelemetrySnapshot per model_id.

        Models with no telemetry at all are not included in the result dict.
        Callers should treat missing keys as "no telemetry" (same as expired).
        """
        async with session_factory() as session:
            # Subquery: max measured_at per model
            from sqlalchemy import func
            subq = (
                select(
                    ModelTelemetryORM.model_id,
                    func.max(ModelTelemetryORM.measured_at).label("max_measured_at"),
                )
                .group_by(ModelTelemetryORM.model_id)
                .subquery()
            )
            result = await session.execute(
                select(ModelTelemetryORM).join(
                    subq,
                    (ModelTelemetryORM.model_id == subq.c.model_id)
                    & (ModelTelemetryORM.measured_at == subq.c.max_measured_at),
                )
            )
            rows = result.scalars().all()

        snapshots: dict[str, TelemetrySnapshot] = {}
        expired_ids: list[str] = []
        unknown_ids: list[str] = []
        for row in rows:
            staleness = _classify_staleness(row.measured_at)
            if staleness == "expired":
                expired_ids.append(row.model_id)
            elif staleness == "unknown":
                unknown_ids.append(row.model_id)
            snapshots[row.model_id] = TelemetrySnapshot(
                model_id=row.model_id,
                measured_at=row.measured_at,
                latency_p50_ms=row.latency_p50_ms,
                is_healthy=row.is_healthy,
                consecutive_failures=row.consecutive_failures or 0,
                staleness=staleness,
            )

        # Emit one summary log per refresh instead of one line per model.
        # This prevents log flooding during telemetry outages (e.g. probes stopped
        # for maintenance) where all 55 models could expire simultaneously.
        if expired_ids:
            log.warning(
                "telemetry_expired",
                count=len(expired_ids),
                model_ids=expired_ids,
            )
        if unknown_ids:
            log.warning(
                "telemetry_stale_warning",
                count=len(unknown_ids),
                model_ids=unknown_ids,
            )

        return snapshots
