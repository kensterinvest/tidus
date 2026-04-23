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
    from tidus.db.repositories.registry_repo import get_active_revision
    from tidus.registry.pipeline import RegistryPipeline
    from tidus.reporting.landing_updater import LandingPageUpdater
    from tidus.reporting.pricing_report import PricingReportGenerator
    from tidus.reporting.subscribers import ReportDelivery, load_subscribers
    from tidus.sync.pricing.hardcoded_source import HardcodedSource

    print(f"[weekly_full_sync] {date.today()}")

    # ── Step 1: DB setup ──────────────────────────────────────────────────────
    print("[1/5] Running price sync pipeline...")
    await create_tables()
    sf = get_session_factory()

    # ── Step 2: Price sync → new DB revision (or detect no changes) ───────────
    pipeline = RegistryPipeline(sf, registry=None)
    result = await pipeline.run_price_sync_cycle([HardcodedSource()])

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
    print("[2/5] Writing weekly snapshot...")
    rows = await pipeline.write_weekly_snapshot(active_revision_id)
    print(f"       {rows} snapshot rows written")

    # ── Step 4: Generate pricing report (md + html) ───────────────────────────
    print("[3/5] Generating pricing report...")
    generator = PricingReportGenerator(sf)
    report = await generator.generate(revision_id=active_revision_id)

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"pricing-{report.report_date}.md"
    html_path = output_dir / f"pricing-{report.report_date}.html"
    md_path.write_text(report.markdown, encoding="utf-8")
    html_path.write_text(report.html, encoding="utf-8")
    print(f"       {md_path}")
    print(f"       {html_path}")

    # ── Step 5: Deliver to subscribers ────────────────────────────────────────
    print("[4/5] Delivering report to subscribers...")
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

    # ── Step 6: Regenerate index.html + push to GitHub ────────────────────────
    print("[5/5] Updating landing page + pushing to GitHub...")
    updater = LandingPageUpdater()
    ok = await updater.update(sf)
    print(f"       Landing update: {'success' if ok else 'failed'}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
