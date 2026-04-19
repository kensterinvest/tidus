#!/usr/bin/env python3
"""Analyze whether E1 (Presidio PERSON alone) catches each of the 14
IRR-adjudicated confidential flips. This probes the architectural claim:
the cheap cheap tier catches entity-bearing confidentials, the LLM tier
catches topic-bearing ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

flips = [
    json.loads(line)
    for line in (ROOT / "tests" / "classification" / "label_overrides_irr.jsonl")
    .read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(f"Total IRR flips: {len(flips)}")

pool_chunks: dict[str, str] = {}
for pf in (ROOT / "tests" / "classification" / "pool_chunks").glob("pool_*.jsonl"):
    for line in pf.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            pool_chunks[r["id"]] = r["text"]
print(f"pool_chunks rows: {len(pool_chunks)}")
in_pool = sum(1 for f in flips if f["id"] in pool_chunks)
print(f"Of {len(flips)} flips, in ensemble pool_chunks: {in_pool}")

from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()

print()
print(f"{'id':<42} {'len':>5}  {'status':<7} {'entities'}")
print("-" * 100)
e1_caught = 0
e1_missed = 0
miss_rows: list[dict] = []

for f in flips:
    rid = f["id"]
    if rid not in pool_chunks:
        print(f"{rid:<42}        NOT-IN-POOL")
        continue
    text = pool_chunks[rid]
    results = analyzer.analyze(text=text[:3000], language="en")
    entities = [r.entity_type for r in results]
    has_person = "PERSON" in entities
    entity_summary = ",".join(sorted(set(entities))[:5]) or "(none)"
    status = "CAUGHT " if has_person else "MISS   "
    print(f"{rid:<42} {len(text):>5}  {status} {entity_summary}")
    if has_person:
        e1_caught += 1
    else:
        e1_missed += 1
        miss_rows.append({"id": rid, "len": len(text), "preview": text[:180]})

print()
print(f"E1 catches: {e1_caught} of {in_pool}")
print(f"E1 misses:  {e1_missed} of {in_pool}")
print()
print("=== E1-miss previews (candidates for Tier-5 LLM escalation) ===")
for m in miss_rows:
    print(f"  [{m['id'][:20]}...] len={m['len']}  preview: {m['preview']!r}")
