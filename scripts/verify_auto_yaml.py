#!/usr/bin/env python3
"""One-off: run ClaudeDiscoveryVerifier over config/models.auto.yaml.

Reason this exists:
  The Sun 2026-05-17 sync auto-promoted 166 models from OpenRouter. The AI
  verifier was wired in (commit 936f2b1) but the GHA secret `ANTHROPIC_API_KEY`
  was never successfully set, so verification ran no-op and bogus entries
  like `claude-opus-4.7-fast` (a model Anthropic doesn't ship) slipped into
  the routing catalog. Production isn't live yet, but the next sync needs a
  clean baseline.

Usage:
    # PowerShell:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    uv run python scripts/verify_auto_yaml.py

    # Bash:
    ANTHROPIC_API_KEY="sk-ant-..." uv run python scripts/verify_auto_yaml.py

What it does:
  1. Loads config/models.auto.yaml (the file written by AutoPromoter).
  2. Builds a DiscoveryCandidate per entry. Since auto.yaml doesn't carry
     openrouter_id or display_name (they're discovery-time-only fields,
     dropped at YAML write), the verifier gets `model_id` as a stand-in —
     enough context to judge plausibility on name + vendor + price.
  3. Calls ClaudeDiscoveryVerifier.verify(...).
  4. Rewrites config/models.auto.yaml with ONLY the accepted entries.
  5. Prints accepted/rejected counts + each rejection's reasoning.

Safety:
  Fail-open: if the API key is missing or Claude is unreachable, the file
  is left UNCHANGED. The script never silently keeps something Claude
  rejected, and never silently drops everything because of an outage.

This is a one-off — once tonight's prune lands and the GHA secret is
properly set, future syncs run the verifier inline and this script
becomes redundant.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> int:
    from tidus.settings import get_settings
    from tidus.sync.ai_verifier import (
        ClaudeDiscoveryVerifier,
        DiscoveryCandidate,
    )

    settings = get_settings()
    auto_path = Path(settings.auto_promote_yaml_path)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment.", file=sys.stderr)
        print(
            "       Set it via:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'  (PowerShell)",
            file=sys.stderr,
        )
        print(
            "                or  ANTHROPIC_API_KEY='sk-ant-...' uv run python scripts/verify_auto_yaml.py",
            file=sys.stderr,
        )
        return 1

    if not auto_path.exists():
        print(f"ERROR: {auto_path} not found — nothing to verify.", file=sys.stderr)
        return 1

    raw = yaml.safe_load(auto_path.read_text(encoding="utf-8")) or {}
    entries: list[dict] = raw.get("models", []) or []
    if not entries:
        print(f"[verify_auto_yaml] {auto_path} is empty. Nothing to do.")
        return 0

    print(f"[verify_auto_yaml] loaded {len(entries)} entries from {auto_path}")

    candidates: list[DiscoveryCandidate] = []
    entries_by_id: dict[str, dict] = {}
    for e in entries:
        mid = e.get("model_id", "")
        if not mid:
            continue
        entries_by_id[mid] = e
        candidates.append(
            DiscoveryCandidate(
                model_id=mid,
                vendor=e.get("vendor", "") or "",
                # auto.yaml dropped these fields — pass model_id as a usable stand-in
                # so Claude still has a non-empty identifier to reason about.
                openrouter_id=mid,
                display_name=e.get("display_name", None),
                input_price_per_1m=float(e.get("input_price", 0)) * 1000,
                output_price_per_1m=float(e.get("output_price", 0)) * 1000,
            )
        )

    verifier = ClaudeDiscoveryVerifier(
        api_key=api_key,
        model=settings.ai_verify_model,
    )
    if not verifier.is_available:
        print("ERROR: verifier reports unavailable — key likely empty.", file=sys.stderr)
        return 1

    print(f"[verify_auto_yaml] calling {settings.ai_verify_model} on {len(candidates)} candidates...")
    verdict = await verifier.verify(candidates)

    if verdict.skipped:
        print(
            f"WARN: verifier skipped (reason: {verdict.skipped_reason}). "
            f"File left unchanged. Try again or check the API key.",
            file=sys.stderr,
        )
        return 1

    print()
    print(f"  accepted: {len(verdict.accepted)}")
    print(f"  rejected: {len(verdict.rejected)}")
    print()

    if verdict.rejected:
        print("Rejected models (will be removed):")
        for r in verdict.rejected:
            print(f"  - {r.candidate.model_id}  ({r.candidate.vendor})")
            print(f"      → {r.reasoning}")
        print()

    accepted_ids = {a.model_id for a in verdict.accepted}
    pruned_entries = [
        entries_by_id[mid] for mid in accepted_ids if mid in entries_by_id
    ]
    # Preserve original ordering for diff readability
    pruned_entries.sort(key=lambda e: e.get("model_id", ""))

    header = (
        "# Tidus auto-promoted model catalog — DO NOT EDIT BY HAND.\n"
        "# Generated by tidus/sync/auto_promote.py on each weekly sync.\n"
        "# Pruned by scripts/verify_auto_yaml.py on 2026-05-17 using\n"
        "# ClaudeDiscoveryVerifier — entries Claude judged implausible were\n"
        "# removed.\n"
    )
    tmp = auto_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(
            {"models": pruned_entries},
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    tmp.replace(auto_path)

    print(
        f"[verify_auto_yaml] wrote {len(pruned_entries)} verified entries to {auto_path}"
    )
    print("[verify_auto_yaml] done.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
