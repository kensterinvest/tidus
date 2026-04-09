"""Drift detectors — Phase 4.

Each detector is responsible for one type of behavioural divergence:

  LatencyDriftDetector     — measured P50 vs catalog P50 ratio
  ContextDriftDetector     — context overflow rate from cost_records
  TokenizationDriftDetector — avg token_delta_pct from model_telemetry
  PriceDriftDetector       — change frequency / deviation from price_change_log

All detectors share the same interface:
  async def detect(session_factory, specs, active_revision_id) -> list[DriftDetection]

Each returns a flat list of DriftDetection objects (one per model per drift type
detected). Callers with severity=None are filtered at construction time — only
warning and critical entries are returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import structlog
from sqlalchemy import func, select

from tidus.db.engine import CostRecordORM, PriceChangeLogORM
from tidus.db.registry_orm import ModelTelemetryORM
from tidus.models.model_registry import ModelSpec

log = structlog.get_logger(__name__)


@dataclass
class DriftDetection:
    """A single drift observation for one model."""

    model_id: str
    drift_type: str                              # latency | context | tokenization | price
    severity: Literal["warning", "critical"]
    metric_value: float
    threshold_value: float                       # the threshold that was exceeded
    active_revision_id: str = ""                 # set by DriftEngine at detection time


# ── LatencyDriftDetector ──────────────────────────────────────────────────────

class LatencyDriftDetector:
    """Detects when fresh measured P50 latency deviates too far from catalog P50.

    Computes the ratio: measured_p50 / catalog_p50.
    Compares the most recent fresh telemetry row against each model's base spec.
    """

    def __init__(
        self,
        warning_ratio: float = 1.5,
        critical_ratio: float = 2.5,
    ) -> None:
        self._warning = warning_ratio
        self._critical = critical_ratio

    async def detect(
        self,
        session_factory,
        specs: list[ModelSpec],
        active_revision_id: str = "",
    ) -> list[DriftDetection]:
        cutoff = datetime.now(UTC) - timedelta(hours=24)  # fresh window

        # Fetch the most recent latency reading per model
        async with session_factory() as session:
            subq = (
                select(
                    ModelTelemetryORM.model_id,
                    func.max(ModelTelemetryORM.measured_at).label("max_ts"),
                )
                .where(
                    ModelTelemetryORM.measured_at >= cutoff,
                    ModelTelemetryORM.latency_p50_ms.is_not(None),
                )
                .group_by(ModelTelemetryORM.model_id)
                .subquery()
            )
            result = await session.execute(
                select(ModelTelemetryORM).join(
                    subq,
                    (ModelTelemetryORM.model_id == subq.c.model_id)
                    & (ModelTelemetryORM.measured_at == subq.c.max_ts),
                )
            )
            rows = {r.model_id: r for r in result.scalars().all()}

        detections: list[DriftDetection] = []
        for spec in specs:
            row = rows.get(spec.model_id)
            if row is None or row.latency_p50_ms is None:
                continue
            if spec.latency_p50_ms == 0:
                continue

            ratio = row.latency_p50_ms / spec.latency_p50_ms
            severity: str | None = None
            threshold = self._warning
            if ratio >= self._critical:
                severity = "critical"
                threshold = self._critical
            elif ratio >= self._warning:
                severity = "warning"
                threshold = self._warning

            if severity:
                detections.append(DriftDetection(
                    model_id=spec.model_id,
                    drift_type="latency",
                    severity=severity,  # type: ignore[arg-type]
                    metric_value=round(ratio, 3),
                    threshold_value=threshold,
                    active_revision_id=active_revision_id,
                ))
                log.warning(
                    "latency_drift_detected",
                    model_id=spec.model_id,
                    ratio=round(ratio, 3),
                    severity=severity,
                )

        return detections


# ── ContextDriftDetector ──────────────────────────────────────────────────────

class ContextDriftDetector:
    """Detects when requests approach the model's context window too often.

    Computes the overflow rate: fraction of requests in the lookback window where
    input_tokens >= max_context * 0.9 (within 10% of the limit). A high overflow
    rate indicates that users are consistently hitting the model's context limit,
    which causes truncation errors and routing quality degradation.

    warning_rate=0.05 means 5% of requests are near the limit.
    critical_rate=0.15 means 15% of requests are near the limit.
    """

    _OVERFLOW_FRACTION = 0.9  # requests using >= 90% of context window count as overflow

    def __init__(
        self,
        warning_rate: float = 0.05,
        critical_rate: float = 0.15,
        lookback_days: int = 7,
    ) -> None:
        self._warning = warning_rate
        self._critical = critical_rate
        self._lookback = lookback_days

    async def detect(
        self,
        session_factory,
        specs: list[ModelSpec],
        active_revision_id: str = "",
    ) -> list[DriftDetection]:
        cutoff = datetime.now(UTC) - timedelta(days=self._lookback)
        spec_by_id = {s.model_id: s for s in specs}
        model_ids = set(spec_by_id.keys())

        # Fetch raw per-request token counts for the window; post-process per model
        # using each model's max_context threshold (varies per model, can't be in SQL).
        async with session_factory() as session:
            result = await session.execute(
                select(CostRecordORM.model_id, CostRecordORM.input_tokens)
                .where(
                    CostRecordORM.timestamp >= cutoff,
                    CostRecordORM.model_id.in_(model_ids),
                )
            )
            records = result.all()

        # Group by model_id
        token_counts: dict[str, list[int]] = {}
        for r in records:
            token_counts.setdefault(r.model_id, []).append(r.input_tokens)

        # Note: models with no requests in the lookback window produce no token_counts
        # entry and are not checked here.  This is intentional — context overflow
        # cannot be measured without traffic.  The LatencyDriftDetector has the same
        # property.  Models reactivated after downtime will appear again once they
        # accumulate cost_records.
        detections: list[DriftDetection] = []
        for model_id, counts in token_counts.items():
            spec = spec_by_id.get(model_id)
            if spec is None or not counts or spec.max_context == 0:
                continue

            overflow_threshold = spec.max_context * self._OVERFLOW_FRACTION
            overflow_count = sum(1 for t in counts if t >= overflow_threshold)
            rate = overflow_count / len(counts)

            severity: str | None = None
            threshold = self._warning
            if rate >= self._critical:
                severity = "critical"
                threshold = self._critical
            elif rate >= self._warning:
                severity = "warning"
                threshold = self._warning

            if severity:
                detections.append(DriftDetection(
                    model_id=model_id,
                    drift_type="context",
                    severity=severity,  # type: ignore[arg-type]
                    metric_value=round(rate, 4),
                    threshold_value=threshold,
                    active_revision_id=active_revision_id,
                ))
                log.warning(
                    "context_drift_detected",
                    model_id=model_id,
                    rate=round(rate, 4),
                    severity=severity,
                )

        return detections


# ── TokenizationDriftDetector ─────────────────────────────────────────────────

class TokenizationDriftDetector:
    """Detects when avg token_delta_pct in telemetry exceeds thresholds.

    token_delta_pct = (actual_tokens - estimated_tokens) / estimated_tokens.
    Aggregates recent telemetry rows over `lookback_days`.
    """

    def __init__(
        self,
        warning_threshold: float = 0.25,
        critical_threshold: float = 0.50,
        lookback_days: int = 7,
    ) -> None:
        self._warning = warning_threshold
        self._critical = critical_threshold
        self._lookback = lookback_days

    async def detect(
        self,
        session_factory,
        specs: list[ModelSpec],
        active_revision_id: str = "",
    ) -> list[DriftDetection]:
        cutoff = datetime.now(UTC) - timedelta(days=self._lookback)
        model_ids = {s.model_id for s in specs}

        async with session_factory() as session:
            result = await session.execute(
                select(
                    ModelTelemetryORM.model_id,
                    func.avg(ModelTelemetryORM.token_delta_pct).label("avg_delta"),
                )
                .where(
                    ModelTelemetryORM.measured_at >= cutoff,
                    ModelTelemetryORM.token_delta_pct.is_not(None),
                    ModelTelemetryORM.model_id.in_(model_ids),
                )
                .group_by(ModelTelemetryORM.model_id)
            )
            rows = {r.model_id: r.avg_delta for r in result.all()}

        detections: list[DriftDetection] = []
        for model_id, avg_delta in rows.items():
            if avg_delta is None:
                continue
            abs_delta = abs(avg_delta)

            severity: str | None = None
            threshold = self._warning
            if abs_delta >= self._critical:
                severity = "critical"
                threshold = self._critical
            elif abs_delta >= self._warning:
                severity = "warning"
                threshold = self._warning

            if severity:
                detections.append(DriftDetection(
                    model_id=model_id,
                    drift_type="tokenization",
                    severity=severity,  # type: ignore[arg-type]
                    metric_value=round(avg_delta, 4),
                    threshold_value=threshold,
                    active_revision_id=active_revision_id,
                ))
                log.warning(
                    "tokenization_drift_detected",
                    model_id=model_id,
                    avg_delta=round(avg_delta, 4),
                    severity=severity,
                )

        return detections


# ── PriceDriftDetector ────────────────────────────────────────────────────────

class PriceDriftDetector:
    """Detects rapid price churn or large deviations in price_change_log.

    Two signals:
    1. Change frequency: > max_changes_30d changes in last 30 days → warning
    2. Magnitude: latest change delta_pct > deviation_warning → warning
    """

    def __init__(
        self,
        max_changes_30d: int = 3,
        deviation_warning: float = 0.15,
        lookback_days: int = 30,
    ) -> None:
        self._max_changes = max_changes_30d
        self._deviation = deviation_warning
        self._lookback = lookback_days

    async def detect(
        self,
        session_factory,
        specs: list[ModelSpec],
        active_revision_id: str = "",
    ) -> list[DriftDetection]:
        cutoff = datetime.now(UTC) - timedelta(days=self._lookback)
        model_ids = {s.model_id for s in specs}

        async with session_factory() as session:
            # Count changes per model
            count_result = await session.execute(
                select(
                    PriceChangeLogORM.model_id,
                    func.count(PriceChangeLogORM.id).label("change_count"),
                    func.max(func.abs(PriceChangeLogORM.delta_pct)).label("max_delta"),
                )
                .where(
                    PriceChangeLogORM.detected_at >= cutoff,
                    PriceChangeLogORM.model_id.in_(model_ids),
                )
                .group_by(PriceChangeLogORM.model_id)
            )
            rows = {r.model_id: (r.change_count, r.max_delta) for r in count_result.all()}

        detections: list[DriftDetection] = []
        for model_id, (change_count, max_delta) in rows.items():
            # Churn signal
            if change_count > self._max_changes:
                detections.append(DriftDetection(
                    model_id=model_id,
                    drift_type="price",
                    severity="warning",
                    metric_value=float(change_count),
                    threshold_value=float(self._max_changes),
                    active_revision_id=active_revision_id,
                ))
                log.warning(
                    "price_drift_churn_detected",
                    model_id=model_id,
                    change_count=change_count,
                )

            # Magnitude signal (only if not already flagged above)
            elif max_delta is not None and max_delta >= self._deviation:
                detections.append(DriftDetection(
                    model_id=model_id,
                    drift_type="price",
                    severity="warning",
                    metric_value=round(float(max_delta), 4),
                    threshold_value=self._deviation,
                    active_revision_id=active_revision_id,
                ))
                log.warning(
                    "price_drift_magnitude_detected",
                    model_id=model_id,
                    max_delta=round(float(max_delta), 4),
                )

        return detections
