"""DriftEngine — orchestrates all drift detectors and applies automated remediation.

Run every 5 minutes (same interval as health probe) by TidusScheduler.

Detection flow:
  1. Run all 4 detectors concurrently against DB
  2. For each new detection: write a ModelDriftEventORM row
  3. Remediation:
     - warning  → no override created; Tier A probe priority set via in-memory flag
     - critical → auto-create hard_disable_model override (created_by='drift_engine')
  4. Recovery check: for each model with an active drift_engine override, check the
     last 3 telemetry rows. If all 3 are healthy → deactivate override + resolve event.

All actions are non-fatal: exceptions are logged and the engine continues.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select

from tidus.auth.middleware import TokenPayload
from tidus.db.registry_orm import ModelDriftEventORM, ModelOverrideORM, ModelTelemetryORM
from tidus.db.repositories.registry_repo import get_open_drift_events
from tidus.models.registry_models import CreateOverrideRequest
from tidus.observability.registry_metrics import DRIFT_EVENTS
from tidus.sync.drift.detectors import (
    ContextDriftDetector,
    DriftDetection,
    LatencyDriftDetector,
    PriceDriftDetector,
    TokenizationDriftDetector,
)
from tidus.utils.yaml_loader import load_yaml

log = structlog.get_logger(__name__)

# Synthetic actor for audit trail entries written by the drift engine.
_DRIFT_ACTOR = TokenPayload(
    sub="drift_engine",
    team_id="system",
    role="admin",
    permissions=[],
    raw_claims={},
)

_RECOVERY_HEALTHY_CYCLES = 3   # consecutive healthy probes needed to auto-resolve critical
_WARNING_AUTO_RESOLVE_HOURS = 72  # resolve open warning events not detected in current cycle


class DriftEngine:
    """Runs all drift detectors, writes events, and applies automated remediation."""

    def __init__(
        self,
        session_factory,
        registry=None,
        override_manager=None,
        policies_path: str = "config/policies.yaml",
    ) -> None:
        self._sf = session_factory
        self._registry = registry
        self._override_manager = override_manager
        self._policies_path = policies_path

    async def run(self) -> list[DriftDetection]:
        """Run all detectors and apply remediation. Returns all detections."""
        if self._registry is None:
            return []

        try:
            raw = load_yaml(self._policies_path)
            drift_cfg = raw.get("drift", {})
        except Exception:
            drift_cfg = {}

        specs = self._registry.list_enabled()
        if not specs:
            return []

        active_revision_id = getattr(self._registry, "active_revision_id", "")

        # Build detectors from config
        lat_cfg = drift_cfg.get("latency", {})
        ctx_cfg = drift_cfg.get("context", {})
        tok_cfg = drift_cfg.get("tokenization", {})
        pri_cfg = drift_cfg.get("price", {})

        detectors = [
            LatencyDriftDetector(
                warning_ratio=lat_cfg.get("warning_ratio", 1.5),
                critical_ratio=lat_cfg.get("critical_ratio", 2.5),
            ),
            ContextDriftDetector(
                warning_rate=ctx_cfg.get("warning_rate", 0.05),
                critical_rate=ctx_cfg.get("critical_rate", 0.15),
                lookback_days=ctx_cfg.get("lookback_days", 7),
            ),
            TokenizationDriftDetector(
                warning_threshold=tok_cfg.get("warning_threshold", 0.25),
                critical_threshold=tok_cfg.get("critical_threshold", 0.50),
                lookback_days=tok_cfg.get("lookback_days", 7),
            ),
            PriceDriftDetector(
                max_changes_30d=pri_cfg.get("max_changes_30d", 3),
                deviation_warning=pri_cfg.get("deviation_warning", 0.15),
                lookback_days=pri_cfg.get("lookback_days", 30),
            ),
        ]

        # Run all detectors concurrently
        results = await asyncio.gather(
            *[d.detect(self._sf, specs, active_revision_id) for d in detectors],
            return_exceptions=True,
        )

        all_detections: list[DriftDetection] = []
        for r in results:
            if isinstance(r, Exception):
                log.error("drift_detector_failed", error=str(r))
            else:
                all_detections.extend(r)

        if all_detections:
            await self._process_detections(all_detections)

        # Recovery check (auto-deactivates critical-drift overrides after 3 healthy probes)
        await self._check_recovery(specs)

        # Auto-resolve warning events that weren't re-detected in this cycle
        # and are older than _WARNING_AUTO_RESOLVE_HOURS.
        detected_keys = {(d.model_id, d.drift_type) for d in all_detections}
        await self._auto_resolve_stale_warnings(detected_keys)

        return all_detections

    async def _process_detections(self, detections: list[DriftDetection]) -> None:
        """Write new drift events and apply remediation for critical detections."""
        existing_open = await get_open_drift_events(self._sf)
        existing_key = {(e.model_id, e.drift_type) for e in existing_open}

        for detection in detections:
            key = (detection.model_id, detection.drift_type)
            if key in existing_key:
                # Already have an open event for this model+type — skip
                continue

            # Write new drift event
            event_id = str(uuid.uuid4())
            try:
                async with self._sf() as session:
                    session.add(ModelDriftEventORM(
                        id=event_id,
                        model_id=detection.model_id,
                        drift_type=detection.drift_type,
                        severity=detection.severity,
                        metric_value=detection.metric_value,
                        threshold_value=detection.threshold_value,
                        drift_status="open",
                        active_revision_id=detection.active_revision_id or None,
                    ))
                    await session.commit()
                DRIFT_EVENTS.labels(
                    model_id=detection.model_id,
                    drift_type=detection.drift_type,
                    severity=detection.severity,
                ).inc()
                log.warning(
                    "drift_event_created",
                    model_id=detection.model_id,
                    drift_type=detection.drift_type,
                    severity=detection.severity,
                    event_id=event_id,
                )
            except Exception as exc:
                log.error("drift_event_write_failed", error=str(exc))
                continue

            # Critical: auto-create hard_disable_model override
            if detection.severity == "critical" and self._override_manager is not None:
                await self._auto_disable(detection, event_id)

    async def _auto_disable(self, detection: DriftDetection, event_id: str) -> None:
        """Create a hard_disable_model override for a critically drifting model."""
        try:
            request = CreateOverrideRequest(
                override_type="hard_disable_model",
                scope="global",
                model_id=detection.model_id,
                payload={},
                justification=(
                    f"drift_engine auto-disable: {detection.drift_type} drift "
                    f"(metric={detection.metric_value:.4f}, "
                    f"threshold={detection.threshold_value:.4f}), "
                    f"event_id={event_id}"
                ),
            )
            override, _ = await self._override_manager.create(request, _DRIFT_ACTOR)
            log.warning(
                "drift_engine_auto_disabled",
                model_id=detection.model_id,
                drift_type=detection.drift_type,
                override_id=override.override_id,
            )
        except Exception as exc:
            log.error("drift_auto_disable_failed", model_id=detection.model_id, error=str(exc))

    async def _check_recovery(self, specs) -> None:
        """For models with active drift_engine overrides, check if they've recovered.

        Uses two sessions total (one for overrides, one for all telemetry) to avoid
        the N+1 session pattern of the naive per-model approach.
        """
        if self._override_manager is None:
            return

        try:
            async with self._sf() as session:
                result = await session.execute(
                    select(ModelOverrideORM).where(
                        ModelOverrideORM.is_active == True,  # noqa: E712
                        ModelOverrideORM.created_by == "drift_engine",
                        ModelOverrideORM.override_type == "hard_disable_model",
                    )
                )
                drift_overrides = result.scalars().all()
        except Exception as exc:
            log.error("drift_recovery_check_failed", error=str(exc))
            return

        model_ids = [o.model_id for o in drift_overrides if o.model_id is not None]
        if not model_ids:
            return

        # Single session: query last N telemetry rows for every candidate model.
        recovered: set[str] = set()
        try:
            async with self._sf() as session:
                for model_id in model_ids:
                    result = await session.execute(
                        select(ModelTelemetryORM.is_healthy)
                        .where(ModelTelemetryORM.model_id == model_id)
                        .order_by(ModelTelemetryORM.measured_at.desc())
                        .limit(_RECOVERY_HEALTHY_CYCLES)
                    )
                    recent = result.scalars().all()
                    if len(recent) >= _RECOVERY_HEALTHY_CYCLES and all(recent):
                        recovered.add(model_id)
        except Exception as exc:
            log.error("drift_recovery_check_failed", error=str(exc))
            return

        for override in drift_overrides:
            if override.model_id not in recovered:
                continue
            try:
                await self._override_manager.deactivate(override.override_id, _DRIFT_ACTOR)
                await self._resolve_open_events(override.model_id)
                log.info(
                    "drift_engine_auto_recovered",
                    model_id=override.model_id,
                    override_id=override.override_id,
                )
            except Exception as exc:
                log.error("drift_recovery_action_failed", model_id=override.model_id, error=str(exc))

    async def _auto_resolve_stale_warnings(
        self, currently_detected: set[tuple[str, str]]
    ) -> None:
        """Auto-resolve warning events that were NOT re-detected in this cycle
        and are older than _WARNING_AUTO_RESOLVE_HOURS.

        Without this, warning events accumulate indefinitely — they have no
        auto-disable override to trigger the recovery path.  Critical events
        are handled by _check_recovery(); this method covers warnings only.

        `currently_detected` is a set of (model_id, drift_type) pairs from
        this cycle's detections — those events are deliberately excluded from
        resolution because the condition is still active.
        """
        cutoff = datetime.now(UTC) - timedelta(hours=_WARNING_AUTO_RESOLVE_HOURS)
        try:
            from sqlalchemy import update as sa_update

            async with self._sf() as session:
                result = await session.execute(
                    select(ModelDriftEventORM).where(
                        ModelDriftEventORM.drift_status == "open",
                        ModelDriftEventORM.severity == "warning",
                        ModelDriftEventORM.detected_at < cutoff,
                    )
                )
                stale = result.scalars().all()

            to_resolve = [
                e for e in stale
                if (e.model_id, e.drift_type) not in currently_detected
            ]
            if not to_resolve:
                return

            ids = [e.id for e in to_resolve]
            now = datetime.now(UTC)
            async with self._sf() as session:
                await session.execute(
                    sa_update(ModelDriftEventORM)
                    .where(ModelDriftEventORM.id.in_(ids))
                    .values(drift_status="auto_resolved", resolved_at=now)
                )
                await session.commit()

            log.info(
                "drift_warnings_auto_resolved",
                count=len(ids),
                age_hours=_WARNING_AUTO_RESOLVE_HOURS,
            )
        except Exception as exc:
            log.error("drift_warning_auto_resolve_failed", error=str(exc))

    async def _resolve_open_events(self, model_id: str) -> None:
        """Mark all open drift events for this model as auto_resolved."""
        now = datetime.now(UTC)
        try:
            from sqlalchemy import update

            async with self._sf() as session:
                await session.execute(
                    update(ModelDriftEventORM)
                    .where(
                        ModelDriftEventORM.model_id == model_id,
                        ModelDriftEventORM.drift_status == "open",
                    )
                    .values(drift_status="auto_resolved", resolved_at=now)
                )
                await session.commit()
        except Exception as exc:
            log.error("drift_resolve_events_failed", model_id=model_id, error=str(exc))
