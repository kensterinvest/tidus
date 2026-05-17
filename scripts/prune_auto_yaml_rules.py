#!/usr/bin/env python3
"""Conservative rule-based prune of config/models.auto.yaml.

Backstop for when ClaudeDiscoveryVerifier can't run (no API credits, no
network, no key). Removes entries whose model_id matches patterns that are
known-bogus for the routing path: dated vendor snapshots that get retired,
speculative version names, name shapes the adapter can't translate.

This is intentionally conservative — it errs on the side of KEEPING
ambiguous entries (they can be removed manually) rather than DROPPING ones
that might be real. The downside of a false-positive prune is a magazine
that's 3 entries short; the downside of a false-negative is a 404 in
production. Same fail-safe direction as the AI verifier.

Usage:
    uv run python scripts/prune_auto_yaml_rules.py [--dry-run]

When --dry-run is set, prints the would-be diff and exits without
touching the file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# Patterns that almost-always indicate "this id can't route through Tidus's
# adapters" or "this is a retired/speculative vendor snapshot". Each entry
# is (regex, reason) — the reason gets printed alongside the rejection for
# transparency.
_PRUNE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ── Anthropic Fast Mode IDs use dot-versioned names that Tidus's
    # canonical-id format doesn't match. Per claude-api skill, Opus 4.7
    # doesn't even ship a Fast variant; 4.6 does but its real model_id
    # uses dashes, not dots, in the canonical form.
    (re.compile(r"^claude-opus-4\.\d+-fast$"), "Fast Mode ID with dot-version — wouldn't match Anthropic API"),

    # ── Dated vendor snapshots (YYYY-MM-DD or YYMMDD shape) are routinely
    # retired by the vendor 6-12 months after release; Tidus's hand-curated
    # entries use stable canonical aliases instead. Drop any with a dated suffix.
    (re.compile(r"-\d{4}-\d{2}-\d{2}$"), "Dated snapshot (YYYY-MM-DD) — retires unpredictably"),
    (re.compile(r"-(0[1-9]|1[0-2])\d{2}$"), "Dated snapshot (MMDD suffix like -0613, -2508) — retires unpredictably"),

    # ── Pre-GA xAI versions: grok-4.20 / grok-4.3 / grok-3-mini-fast etc.
    # are not in xAI's GA catalog at any point I've seen.
    (re.compile(r"^grok-4\.\d+"), "Speculative Grok version — not in xAI GA catalog"),
    (re.compile(r"^grok-3-(mini-)?fast$"), "Grok-3 fast variant — not in xAI GA catalog"),

    # ── Distilled/finetuned variants don't route through vendor APIs — they're
    # hosted on third parties OpenRouter brokers but no Tidus adapter targets them.
    (re.compile(r"-distill-"), "Distilled variant — no Tidus adapter for the host"),

    # ── Legacy OpenAI models that have been deprecated or retired:
    (re.compile(r"^gpt-3\.5-turbo($|-(0613|1106|16k|instruct))"),
     "Legacy GPT-3.5 — likely retired or to-be-retired by OpenAI"),
    (re.compile(r"^gpt-4(-0314|-0613|-turbo$|-turbo-2024)"),
     "Legacy GPT-4 dated snapshot — retired or to-be-retired"),
    (re.compile(r"^gpt-4o-(2024|mini-2024)"),
     "Legacy GPT-4o dated snapshot — superseded by undated alias"),
    (re.compile(r"^chatgpt-4o-latest"),
     "ChatGPT-4o-latest — moving target, not stable for routing"),

    # ── Legacy Anthropic models retired or to be retired:
    (re.compile(r"^claude-3-(haiku|opus|sonnet)$"), "Claude 3 — retired or deprecated"),
    (re.compile(r"^claude-3\.5-(haiku|sonnet)"), "Claude 3.5 family — retired"),
    (re.compile(r"^claude-opus-4($|\.[01]$|\.5$)"), "Claude Opus 4.0-4.5 — superseded by 4.6/4.7"),
    (re.compile(r"^claude-sonnet-4($|\.5$)"), "Claude Sonnet 4.0-4.5 — superseded by 4.6"),

    # ── Retired Google Gemma open-weight models (open-weight, not Google's API).
    (re.compile(r"^gemma-2-"), "Gemma 2 — open-weight model, no Google adapter target"),
]


def _should_prune(model_id: str) -> tuple[bool, str]:
    for pat, reason in _PRUNE_PATTERNS:
        if pat.search(model_id):
            return True, reason
    return False, ""


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    from tidus.settings import get_settings

    auto_path = Path(get_settings().auto_promote_yaml_path)
    if not auto_path.exists():
        print(f"ERROR: {auto_path} not found.", file=sys.stderr)
        return 1

    raw = yaml.safe_load(auto_path.read_text(encoding="utf-8")) or {}
    entries: list[dict] = raw.get("models", []) or []
    if not entries:
        print(f"[prune] {auto_path} is empty. Nothing to do.")
        return 0

    keep: list[dict] = []
    pruned: list[tuple[str, str, str]] = []   # (model_id, vendor, reason)

    for e in entries:
        mid = e.get("model_id", "")
        if not mid:
            continue
        should, reason = _should_prune(mid)
        if should:
            pruned.append((mid, e.get("vendor", "?"), reason))
        else:
            keep.append(e)

    print(f"[prune] read {len(entries)} entries from {auto_path}")
    print(f"        keep:   {len(keep)}")
    print(f"        prune:  {len(pruned)}")
    print()

    if pruned:
        # Group by reason for a readable summary
        by_reason: dict[str, list[tuple[str, str]]] = {}
        for mid, vendor, reason in pruned:
            by_reason.setdefault(reason, []).append((mid, vendor))
        print("Pruned entries:")
        for reason, items in sorted(by_reason.items()):
            print(f"  • {reason}  ({len(items)})")
            for mid, vendor in sorted(items):
                print(f"      - {vendor}/{mid}")
        print()

    if dry_run:
        print("[prune] --dry-run: file NOT modified.")
        return 0

    if not pruned:
        print("[prune] nothing to remove; file untouched.")
        return 0

    keep.sort(key=lambda e: e.get("model_id", ""))
    header = (
        "# Tidus auto-promoted model catalog — DO NOT EDIT BY HAND.\n"
        "# Generated by tidus/sync/auto_promote.py on each weekly sync.\n"
        "# Conservatively pruned by scripts/prune_auto_yaml_rules.py on\n"
        "# 2026-05-17 — see _PRUNE_PATTERNS for the exact rules. Run the\n"
        "# AI verifier (scripts/verify_auto_yaml.py) for tighter cleanup\n"
        "# once Anthropic API credits are available.\n"
    )
    tmp = auto_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(
            {"models": keep},
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    tmp.replace(auto_path)
    print(f"[prune] wrote {len(keep)} entries to {auto_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
