"""APScheduler wiring — health probe (5 min) + weekly price sync.

Started from the FastAPI lifespan alongside the app. Both jobs are
non-blocking background tasks; failures are logged but don't crash the server.

Example:
    scheduler = TidusScheduler(registry, session_factory)
    scheduler.start()
    # ... app runs ...
    scheduler.shutdown()
"""

from __future__ import annotations

import asyncio

import structlog

log = structlog.get_logger(__name__)


class TidusScheduler:
    """Wraps APScheduler to run health probes and price sync."""

    def __init__(self, registry, session_factory=None, policies_path: str = "config/policies.yaml") -> None:
        self._registry = registry
        self._session_factory = session_factory
        self._policies_path = policies_path
        self._scheduler = None

    def start(self) -> None:
        """Start background scheduler jobs."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
            from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
            from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import-untyped]
        except ImportError:
            log.warning(
                "scheduler_disabled",
                reason="apscheduler not installed",
                install="pip install apscheduler",
            )
            return

        from tidus.utils.yaml_loader import load_yaml
        raw = load_yaml(self._policies_path)
        health_cfg = raw.get("health", {})
        sync_cfg = raw.get("pricing_sync", {})

        probe_interval = health_cfg.get("probe_interval_seconds", 300)
        sync_day = sync_cfg.get("day_of_week", 6)
        sync_hour = sync_cfg.get("hour_utc", 2)

        self._scheduler = AsyncIOScheduler()

        self._scheduler.add_job(
            self._run_health_probe,
            trigger=IntervalTrigger(seconds=probe_interval),
            id="health_probe",
            name="Model health probe",
            misfire_grace_time=60,
        )

        self._scheduler.add_job(
            self._run_price_sync,
            trigger=CronTrigger(day_of_week=sync_day, hour=sync_hour, timezone="UTC"),
            id="price_sync",
            name="Weekly price sync",
            misfire_grace_time=3600,
        )

        self._scheduler.start()
        log.info(
            "scheduler_started",
            health_probe_interval_seconds=probe_interval,
            price_sync_day=sync_day,
            price_sync_hour_utc=sync_hour,
        )

    def shutdown(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            log.info("scheduler_stopped")

    async def _run_health_probe(self) -> None:
        try:
            from tidus.sync.health_probe import HealthProbe
            probe = HealthProbe(self._registry, self._policies_path)
            results = await probe.run_once()
            healthy = sum(1 for v in results.values() if v)
            total = len(results)
            log.info("health_probe_run", healthy=healthy, total=total)
        except Exception as exc:
            log.error("health_probe_failed", error=str(exc))

    async def _run_price_sync(self) -> None:
        try:
            from tidus.sync.price_sync import run_price_sync
            changes = await run_price_sync(
                self._registry,
                policies_path=self._policies_path,
                session_factory=self._session_factory,
            )
            log.info("price_sync_run", changes_detected=len(changes))
        except Exception as exc:
            log.error("price_sync_failed", error=str(exc))
