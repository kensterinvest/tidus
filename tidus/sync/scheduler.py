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

import structlog

log = structlog.get_logger(__name__)


class TidusScheduler:
    """Wraps APScheduler to run health probes, price sync, and monthly budget resets."""

    def __init__(
        self,
        registry,
        session_factory=None,
        enforcer=None,
        policies_path: str = "config/policies.yaml",
    ) -> None:
        self._registry = registry
        self._session_factory = session_factory
        self._enforcer = enforcer
        self._policies_path = policies_path
        self._scheduler = None

    def start(self) -> None:
        """Start background scheduler jobs."""
        try:
            from apscheduler.schedulers.asyncio import (
                AsyncIOScheduler,  # type: ignore[import-untyped]
            )
            from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
            from apscheduler.triggers.interval import (
                IntervalTrigger,  # type: ignore[import-untyped]
            )
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

        # Monthly budget reset — 1st of each month at 00:05 UTC
        self._scheduler.add_job(
            self._run_monthly_budget_reset,
            trigger=CronTrigger(day=1, hour=0, minute=5, timezone="UTC"),
            id="monthly_budget_reset",
            name="Monthly budget period reset",
            misfire_grace_time=3600,
        )

        # Registry cache refresh — detect revision or override changes every 60s
        self._scheduler.add_job(
            self._run_registry_refresh,
            trigger=IntervalTrigger(seconds=60),
            id="registry_refresh",
            name="Registry cache refresh",
            misfire_grace_time=30,
        )

        # Override expiry — deactivate expired overrides every 15 minutes
        self._scheduler.add_job(
            self._run_override_expiry,
            trigger=IntervalTrigger(seconds=900),
            id="override_expiry",
            name="Override expiry enforcement",
            misfire_grace_time=300,
        )

        # Drift detection — run all 4 detectors every 5 minutes after health probe
        self._scheduler.add_job(
            self._run_drift_engine,
            trigger=IntervalTrigger(seconds=probe_interval),
            id="drift_engine",
            name="Drift detection engine",
            misfire_grace_time=60,
        )

        # Metrics refresh — update registry Gauges every 5 minutes
        self._scheduler.add_job(
            self._run_metrics_update,
            trigger=IntervalTrigger(seconds=probe_interval),
            id="metrics_updater",
            name="Registry metrics updater",
            misfire_grace_time=60,
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
            probe = HealthProbe(
                self._registry,
                self._policies_path,
                session_factory=self._session_factory,
            )
            results = await probe.run_once()
            healthy = sum(1 for v in results.values() if v)
            total = len(results)
            log.info("health_probe_run", healthy=healthy, total=total)
        except Exception as exc:
            log.error("health_probe_failed", error=str(exc))

    async def _run_monthly_budget_reset(self) -> None:
        if self._enforcer is None:
            log.warning("monthly_budget_reset_skipped", reason="no enforcer configured")
            return
        try:
            from tidus.models.budget import BudgetPeriod
            count = await self._enforcer.reset_period(BudgetPeriod.monthly)
            log.info("monthly_budget_reset_run", reset_count=count)
        except Exception as exc:
            log.error("monthly_budget_reset_failed", error=str(exc))

    async def _run_price_sync(self) -> None:
        try:
            from tidus.registry.pipeline import PipelineResult, RegistryPipeline
            from tidus.settings import get_settings
            from tidus.sync.pricing.hardcoded_source import HardcodedSource

            settings = get_settings()
            sources = [HardcodedSource()]

            if settings.tidus_pricing_feed_url:
                from tidus.sync.pricing.feed_source import TidusPricingFeedSource
                sources.append(TidusPricingFeedSource(
                    feed_url=settings.tidus_pricing_feed_url,
                    signing_key=settings.tidus_pricing_feed_signing_key,
                    failure_threshold=settings.pricing_feed_failure_threshold,
                    reset_timeout_seconds=settings.pricing_feed_reset_timeout_seconds,
                ))

            pipeline = RegistryPipeline(self._session_factory, self._registry)
            result = await pipeline.run_price_sync_cycle(
                sources, policies_path=self._policies_path
            )
            if isinstance(result, PipelineResult):
                log.info(
                    "price_sync_run",
                    changes_detected=len(result.changes),
                    revision_id=result.revision_id,
                    sources=result.sources_used,
                )
                # Generate and deliver pricing report to subscribers
                await self._run_pricing_report(result.revision_id)
            else:
                log.info("price_sync_run", changes_detected=0)
        except Exception as exc:
            log.error("price_sync_failed", error=str(exc))

    async def _run_pricing_report(self, revision_id: str) -> None:
        """Generate Tidus AI Model Latest Pricing Report and deliver to subscribers."""
        try:
            from pathlib import Path
            from tidus.reporting.pricing_report import PricingReportGenerator
            from tidus.reporting.subscribers import ReportDelivery, load_subscribers

            generator = PricingReportGenerator(self._session_factory)
            report = await generator.generate(revision_id=revision_id)

            # Write report to reports/ directory
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            report_path = reports_dir / f"pricing-{report.report_date}.md"
            report_path.write_text(report.markdown, encoding="utf-8")
            log.info("pricing_report_written", path=str(report_path))

            # Deliver to subscribers
            subscribers = load_subscribers()
            if subscribers:
                delivery = ReportDelivery()
                n = delivery.deliver(
                    report_markdown=report.markdown,
                    subject=f"Tidus AI Pricing Update — {report.report_date}",
                    subscribers=subscribers,
                    report_html=report.html,
                )
                log.info("pricing_report_delivered", recipients=n)
        except Exception as exc:
            log.error("pricing_report_failed", error=str(exc))

    async def _run_registry_refresh(self) -> None:
        """Check for revision or override changes; rebuild cache if needed."""
        if self._session_factory is None or not hasattr(self._registry, "refresh"):
            return
        try:
            rebuilt = await self._registry.refresh(self._session_factory)
            if rebuilt:
                log.info("registry_refresh_rebuilt")
        except Exception as exc:
            log.error("registry_refresh_failed", error=str(exc))

    async def _run_override_expiry(self) -> None:
        """Deactivate expired overrides and refresh the registry cache."""
        if self._session_factory is None:
            return
        try:
            from tidus.sync.override_expiry import OverrideExpiryJob
            count = await OverrideExpiryJob().run(self._session_factory, self._registry)
            if count:
                log.info("override_expiry_run", deactivated=count)
        except Exception as exc:
            log.error("override_expiry_failed", error=str(exc))

    async def _run_metrics_update(self) -> None:
        """Refresh registry Gauge metrics from DB and in-memory registry."""
        if self._session_factory is None:
            return
        try:
            from tidus.observability.metrics_updater import MetricsUpdater
            await MetricsUpdater().update(self._registry, self._session_factory)
        except Exception as exc:
            log.error("metrics_update_failed", error=str(exc))

    async def _run_drift_engine(self) -> None:
        """Run all drift detectors and apply automated remediation."""
        if self._session_factory is None or self._registry is None:
            return
        try:
            from tidus.sync.drift.engine import DriftEngine
            from tidus.api.deps import get_override_manager
            override_manager = get_override_manager() if self._session_factory else None
            engine = DriftEngine(
                self._session_factory,
                self._registry,
                override_manager=override_manager,
                policies_path=self._policies_path,
            )
            detections = await engine.run()
            if detections:
                log.info("drift_engine_run", detections=len(detections))
        except Exception as exc:
            log.error("drift_engine_failed", error=str(exc))
