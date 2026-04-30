#!/usr/bin/env python3
"""Smoke-test the vendor model discovery against real APIs.

Reads API keys from .env (via Settings), polls each vendor, and prints
what would surface in the next pricing report. Does NOT touch the
discovered_models.json sidecar (uses a tmp path).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> int:
    from tidus.db.engine import create_tables, get_session_factory
    from tidus.db.repositories.registry_repo import (
        get_active_revision,
        get_entries_for_revision,
    )
    from tidus.settings import get_settings
    from tidus.sync.discovery import DiscoveryRunner, build_discovery_sources

    settings = get_settings()
    sources = build_discovery_sources(settings)
    print(f"Sources configured: {[s.source_name for s in sources]}")

    if not sources:
        print("No vendor API keys — nothing to discover.")
        return 1

    await create_tables()
    sf = get_session_factory()
    rev = await get_active_revision(sf)
    if rev is None:
        print("No active registry revision — run weekly_full_sync first.")
        return 1
    entries = await get_entries_for_revision(sf, rev.revision_id)
    registry_ids = {e.model_id for e in entries}
    print(f"Active registry: {len(registry_ids)} models in revision {rev.revision_id[:8]}")

    with tempfile.TemporaryDirectory() as tmp:
        runner = DiscoveryRunner(
            sources,
            state_path=Path(tmp) / "discovered.json",
            registry_model_ids=registry_ids,
        )
        report = await runner.run()

    print()
    print("=" * 60)
    print(f"Sources run:        {report.sources_run}")
    print(f"Sources skipped:    {report.sources_skipped}")
    print(f"Total discovered:   {report.total_discovered}")
    print(f"NEW this run:       {len(report.new_this_run)}")
    print()

    if report.new_this_run:
        print("--- NEWLY DISCOVERED (not in active registry) ---")
        for m in report.new_this_run:
            name = m.display_name or "—"
            print(f"  {m.vendor:12s} canonical={m.model_id:35s} vendor_id={m.vendor_id:35s} ({name})")
    else:
        print("(no new models — vendor catalogs match registry)")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
