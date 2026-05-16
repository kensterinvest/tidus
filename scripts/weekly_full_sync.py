#!/usr/bin/env python3
"""Tidus full pricing sync — standalone orchestration script.

Runs outside the FastAPI server (no canary probes, no APScheduler).
Invoked by the GitHub Actions workflow `.github/workflows/weekly-sync.yml`
on Sundays and Wednesdays at 02:00 UTC. (File name kept as `weekly_full_sync.py`
for git-history continuity — the workflow file and cron cadence are the
source of truth for when this actually fires.)

Usage:
    TIDUS_CANARY_SAMPLE_SIZE=0 uv run python scripts/weekly_full_sync.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date
from pathlib import Path

# Must be set before any tidus imports to skip live canary probes.
os.environ.setdefault("TIDUS_CANARY_SAMPLE_SIZE", "0")

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> int:
    from tidus.db.engine import create_tables, get_session_factory
    from tidus.db.repositories.registry_repo import (
        get_active_revision,
        get_entries_for_revision,
    )
    from tidus.registry.pipeline import RegistryPipeline
    from tidus.reporting.landing_updater import LandingPageUpdater
    from tidus.reporting.pricing_report import PricingReportGenerator
    from tidus.reporting.subscribers import ReportDelivery, load_subscribers
    from tidus.settings import get_settings
    from tidus.sync.discovery import DiscoveryRunner, build_discovery_sources
    from tidus.sync.pricing.base import PricingSource
    from tidus.sync.pricing.hardcoded_source import HardcodedSource
    from tidus.sync.pricing.openrouter_source import OpenRouterPricingSource

    print(f"[weekly_full_sync] {date.today()}")

    # ── Step 1: DB setup ──────────────────────────────────────────────────────
    print("[1/6] Running price sync pipeline...")
    await create_tables()
    sf = get_session_factory()

    # ── Step 2: Price sync → new DB revision (or detect no changes) ───────────
    # HardcodedSource is the verified-baseline anchor; OpenRouter provides a
    # live "second opinion" so consensus.py can catch real vendor price moves
    # between manual hardcoded-table edits. Both fail-safe to [] on network
    # error — the pipeline tolerates any subset being unavailable.
    settings = get_settings()
    pricing_sources: list[PricingSource] = [HardcodedSource()]
    if settings.openrouter_enabled:
        pricing_sources.append(
            OpenRouterPricingSource(
                base_url=settings.openrouter_base_url,
                timeout_seconds=settings.openrouter_request_timeout_seconds,
            )
        )
    pipeline = RegistryPipeline(sf, registry=None)
    result = await pipeline.run_price_sync_cycle(pricing_sources)

    if result is not None:
        active_revision_id = result.revision_id
        n_changes = len(result.changes)
        print(f"       Revision created: {active_revision_id} ({n_changes} changes)")
    else:
        # No price changes — use the currently active revision.
        rev = await get_active_revision(sf)
        if rev is None:
            print("ERROR: No active revision found. Run the seeder first.")
            return 1
        active_revision_id = rev.revision_id
        print(f"       No price changes. Using active revision: {active_revision_id}")

    # ── Step 3: Weekly snapshot (time-series row) ─────────────────────────────
    print("[2/6] Writing weekly snapshot...")
    rows = await pipeline.write_weekly_snapshot(active_revision_id)
    print(f"       {rows} snapshot rows written")

    # ── Step 4: Vendor model discovery ────────────────────────────────────────
    # Polls each vendor's `/v1/models` endpoint to detect new models. Surface
    # only — never auto-routes; promotion still requires a human edit.
    discovery_report = None
    if settings.discovery_enabled:
        print("[3/6] Running vendor model discovery...")
        sources = build_discovery_sources(settings)
        if not sources:
            print("       No vendor API keys configured — discovery skipped.")
        else:
            entries = await get_entries_for_revision(sf, active_revision_id)
            registry_ids = {e.model_id for e in entries}
            runner = DiscoveryRunner(
                sources,
                state_path=Path(settings.discovery_state_path),
                registry_model_ids=registry_ids,
            )
            discovery_report = await runner.run()
            print(
                f"       Sources run: {len(discovery_report.sources_run)}, "
                f"new this run: {len(discovery_report.new_this_run)}, "
                f"pending review: {len(discovery_report.pending_review)}"
            )
    else:
        print("[3/6] Discovery disabled (settings.discovery_enabled=False)")

    # ── Drift alarm: did the active revision sit unchanged for too long? ─────
    # Triggers only when a live source (OpenRouter or external feed) is in
    # play this run — a stuck revision with only HardcodedSource is expected
    # behavior, not a market signal. Reads the threshold from policies.yaml so
    # operators can tune it without code changes.
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from tidus.utils.yaml_loader import load_yaml

    drift_alarm_days: int | None = None
    if result is None and settings.openrouter_enabled:
        policies = load_yaml(settings.policies_config_path)
        threshold_days = int(policies.get("pricing_sync", {}).get("drift_alarm_days", 21))
        active_rev = await get_active_revision(sf)
        if active_rev and active_rev.activated_at:
            activated_at = active_rev.activated_at
            if activated_at.tzinfo is None:
                activated_at = activated_at.replace(tzinfo=_UTC)
            days_stale = (_dt.now(_UTC) - activated_at).days
            if days_stale >= threshold_days:
                drift_alarm_days = days_stale
                print(
                    f"       ⚠️ Drift alarm: active revision unchanged for "
                    f"{days_stale} days (threshold: {threshold_days})."
                )

    # ── Step 5: Generate pricing report (md + html) ───────────────────────────
    print("[4/6] Generating pricing report...")
    generator = PricingReportGenerator(sf)
    report = await generator.generate(
        revision_id=active_revision_id,
        discovery_report=discovery_report,
        drift_alarm_days=drift_alarm_days,
    )

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"pricing-{report.report_date}.md"
    html_path = output_dir / f"pricing-{report.report_date}.html"
    md_path.write_text(report.markdown, encoding="utf-8")
    html_path.write_text(report.html, encoding="utf-8")
    print(f"       {md_path}")
    print(f"       {html_path}")

    # ── Step 6: Deliver to subscribers ────────────────────────────────────────
    print("[5/6] Delivering report to subscribers...")
    subscribers = load_subscribers()
    subject = f"Tidus Pricing Update — {report.report_date}"
    delivery = ReportDelivery()
    delivered = delivery.deliver(
        report_markdown=report.markdown,
        subject=subject,
        subscribers=subscribers,
        report_html=report.html,
    )
    print(f"       Delivered to {delivered}/{len(subscribers)} subscribers")

    # ── Step 7: Regenerate index.html + push to GitHub ────────────────────────
    print("[6/6] Updating landing page + pushing to GitHub...")
    updater = LandingPageUpdater()
    ok = await updater.update(sf)
    print(f"       Landing update: {'success' if ok else 'failed'}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
