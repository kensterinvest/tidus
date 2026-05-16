#!/usr/bin/env python3
"""Tidus full pricing sync — standalone orchestration script.

Runs outside the FastAPI server (no canary probes, no APScheduler).
Invoked by the GitHub Actions workflow `.github/workflows/weekly-sync.yml`
on Sundays and Wednesdays at 02:00 UTC. (File name kept as `weekly_full_sync.py`
for git-history continuity — the workflow file and cron cadence are the
source of truth for when this actually fires.)

Pipeline order:
    1. Discovery    — poll OpenRouter (+ any per-vendor sources with keys)
                      to learn what models exist in the market right now.
    2. Auto-promote — write config/models.auto.yaml so the next steps see
                      newly discovered priced models as part of the catalog.
    3. Price sync   — consensus across HardcodedSource + OpenRouter, may
                      activate a new registry revision. Auto-promoted models
                      enter the DB here because ModelRegistry now merges
                      models.yaml + models.auto.yaml.
    4. Snapshot     — weekly time-series row for trending.
    5. Drift alarm  — flag if the revision hasn't moved despite live data.
    6. Report       — markdown + html magazine.
    7. Subscribers  — Resend email delivery.
    8. Landing      — regen kensterinvest.github.io/tidus/index.html.

Usage:
    TIDUS_CANARY_SAMPLE_SIZE=0 uv run python scripts/weekly_full_sync.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, date, datetime
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
    from tidus.sync.auto_promote import AutoPromoter
    from tidus.sync.discovery import DiscoveryRunner, build_discovery_sources
    from tidus.sync.pricing.base import PricingSource
    from tidus.sync.pricing.hardcoded_source import HardcodedSource
    from tidus.sync.pricing.openrouter_source import OpenRouterPricingSource
    from tidus.utils.yaml_loader import load_yaml

    print(f"[weekly_full_sync] {date.today()}")
    settings = get_settings()

    # ── Step 0: DB setup ──────────────────────────────────────────────────────
    await create_tables()
    sf = get_session_factory()

    # ── Step 1: Vendor model discovery ────────────────────────────────────────
    # Runs FIRST so auto-promote in step 2 can act on the freshest catalog
    # before the pricing pipeline reads merged yaml in step 3.
    discovery_report = None
    if settings.discovery_enabled:
        print("[1/8] Running vendor model discovery...")
        sources = build_discovery_sources(settings)
        if not sources:
            print("       No discovery sources available — skipped.")
        else:
            active_rev = await get_active_revision(sf)
            registry_ids: set[str] = set()
            if active_rev:
                entries = await get_entries_for_revision(sf, active_rev.revision_id)
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
        print("[1/8] Discovery disabled (settings.discovery_enabled=False)")

    # ── Step 2: Auto-promote discovered+priced models into auto.yaml ──────────
    # Writes config/models.auto.yaml. ModelRegistry.load() merges this with
    # models.yaml so the price-sync pipeline in step 3 sees the new entries.
    print("[2/8] Auto-promoting discovered models with live pricing...")
    if discovery_report is not None and settings.auto_promote_enabled:
        hand_curated_raw = load_yaml(settings.models_config_path)
        hand_curated_ids = {
            m.get("model_id") for m in hand_curated_raw.get("models", [])
            if m.get("model_id")
        }
        promoter = AutoPromoter(
            auto_yaml_path=settings.auto_promote_yaml_path,
            enabled=True,
        )
        # All discovered candidates: new_this_run + pending_review + everything
        # currently in_registry. We rebuild the full file each run so an
        # operator's hand promotion (move from auto.yaml → models.yaml)
        # doesn't get clobbered by the next run.
        all_discovered = list(discovery_report.new_this_run) + list(discovery_report.pending_review)
        ap_result = promoter.run(
            discovered=all_discovered,
            hand_curated_ids=hand_curated_ids,
        )
        print(
            f"       Promoted: {len(ap_result.promoted)}, "
            f"already vetted: {ap_result.skipped_known}, "
            f"unknown vendor: {ap_result.skipped_unknown_vendor}, "
            f"no price: {ap_result.skipped_no_price}, "
            f"variant: {ap_result.skipped_variant}"
        )
    else:
        reason = (
            "discovery returned nothing"
            if discovery_report is None
            else "auto_promote_enabled=False"
        )
        print(f"       Skipped ({reason}).")

    # ── Step 3: Price sync → new DB revision (or detect no changes) ───────────
    # HardcodedSource is the verified-baseline anchor; OpenRouter provides
    # live "second opinion" so consensus.py can catch real vendor price
    # moves. Both fail-safe to [] on network error — pipeline tolerates
    # any subset being unavailable.
    print("[3/8] Running price sync pipeline...")
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
        rev = await get_active_revision(sf)
        if rev is None:
            print("ERROR: No active revision found. Run the seeder first.")
            return 1
        active_revision_id = rev.revision_id
        print(f"       No price changes. Using active revision: {active_revision_id}")

    # ── Step 4: Weekly snapshot (time-series row) ────────────────────────────
    print("[4/8] Writing weekly snapshot...")
    rows = await pipeline.write_weekly_snapshot(active_revision_id)
    print(f"       {rows} snapshot rows written")

    # ── Step 5: Drift alarm ──────────────────────────────────────────────────
    # Triggers only when (a) no revision was created and (b) a live source
    # was active. A stuck revision under HardcodedSource-only is expected;
    # under OpenRouter it's a real signal the source is broken or the market
    # genuinely hasn't moved.
    print("[5/8] Computing drift alarm...")
    drift_alarm_days: int | None = None
    if result is None and settings.openrouter_enabled:
        policies = load_yaml(settings.policies_config_path)
        threshold_days = int(policies.get("pricing_sync", {}).get("drift_alarm_days", 21))
        active_rev = await get_active_revision(sf)
        if active_rev and active_rev.activated_at:
            activated_at = active_rev.activated_at
            if activated_at.tzinfo is None:
                activated_at = activated_at.replace(tzinfo=UTC)
            days_stale = (datetime.now(UTC) - activated_at).days
            if days_stale >= threshold_days:
                drift_alarm_days = days_stale
                print(
                    f"       ⚠️ Drift alarm: active revision unchanged for "
                    f"{days_stale} days (threshold: {threshold_days})."
                )

    # ── Step 6: Generate pricing report (md + html) ──────────────────────────
    print("[6/8] Generating pricing report...")
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

    # ── Step 7: Deliver to subscribers ───────────────────────────────────────
    print("[7/8] Delivering report to subscribers...")
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

    # ── Step 8: Regenerate index.html + push to GitHub ───────────────────────
    print("[8/8] Updating landing page + pushing to GitHub...")
    updater = LandingPageUpdater()
    ok = await updater.update(sf)
    print(f"       Landing update: {'success' if ok else 'failed'}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
