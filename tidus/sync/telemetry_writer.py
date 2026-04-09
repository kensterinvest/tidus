"""TelemetryWriter — persists health probe results to model_telemetry.

One row is written per probe observation. Non-fatal: exceptions are logged
but never raised so a DB hiccup cannot crash the health probe.

The persisted rows serve two purposes:
  1. EffectiveRegistry reads the most-recent row per model to apply telemetry
     to the merge layer (latency override, disabled state).
  2. DriftEngine reads historical rows to compute drift metrics.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog

from tidus.db.registry_orm import ModelTelemetryORM

log = structlog.get_logger(__name__)


class TelemetryWriter:
    """Inserts a single ModelTelemetryORM row. Stateless — call write() directly."""

    @staticmethod
    async def write(
        session_factory,
        model_id: str,
        is_healthy: bool,
        latency_ms: float | None = None,
        consecutive_failures: int = 0,
        probe_type: str | None = None,  # "synthetic" | "live"
        measured_at: datetime | None = None,
        context_exceeded_rate: float | None = None,
        token_delta_pct: float | None = None,
        source: str = "health_probe",  # "health_probe" | "request_log"
    ) -> None:
        """Insert one telemetry row. Never raises."""
        try:
            now = measured_at or datetime.now(UTC)
            latency_int = int(latency_ms) if latency_ms is not None else None

            async with session_factory() as session:
                session.add(ModelTelemetryORM(
                    id=str(uuid.uuid4()),
                    model_id=model_id,
                    measured_at=now,
                    latency_p50_ms=latency_int,
                    is_healthy=is_healthy,
                    consecutive_failures=consecutive_failures,
                    context_exceeded_rate=context_exceeded_rate,
                    token_delta_pct=token_delta_pct,
                    source=source,
                    probe_type=probe_type,
                ))
                await session.commit()
        except Exception as exc:
            log.error(
                "telemetry_write_failed",
                model_id=model_id,
                error=str(exc),
            )
