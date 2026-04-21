#!/usr/bin/env python3
"""P1 Step 1: Score the CURRENT encoder against the 83 IRR confidentials.

This is the "leaked" baseline — the current weights_b/ was trained with IRR
overrides merged in, so recall here is upper-bounded by memorization. Used only
for before/after comparison with a retrained (IRR-excluded) encoder.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = REPO_ROOT / "tests" / "classification" / "pool_chunks"
IRR_PATH = REPO_ROOT / "tests" / "classification" / "label_overrides_irr.jsonl"
WEIGHTS_DIR = REPO_ROOT / "tidus" / "classification" / "weights_b"
MAX_CHARS = 1200


def main() -> int:
    mappings = json.loads((WEIGHTS_DIR / "label_mappings.json").read_text())
    privacies: list[str] = mappings["privacies"]
    conf_idx = privacies.index("confidential")

    irr = []
    with IRR_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            irr.append(json.loads(line))
    print(f"Loaded {len(irr)} IRR override rows")

    pool_text: dict[str, str] = {}
    for pf in sorted(POOL_DIR.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                pool_text[r["id"]] = r["text"]

    rows = [(r["id"], pool_text[r["id"]], r["privacy"]) for r in irr if r["id"] in pool_text]
    missing = len(irr) - len(rows)
    if missing:
        print(f"  WARN: {missing} IRR ids not found in pool")
    print(f"Joined IRR to pool text: {len(rows)} rows")

    true_dist = Counter(p for _, _, p in rows)
    print(f"Ground-truth privacy distribution (IRR): {dict(true_dist)}")

    print(f"\nLoading {mappings.get('embed_model', 'all-MiniLM-L6-v2')}...")
    embed = SentenceTransformer(mappings.get("embed_model", "all-MiniLM-L6-v2"))
    clf_priv = joblib.load(WEIGHTS_DIR / "privacy_head.joblib")

    texts = [t[:MAX_CHARS] for _, t, _ in rows]
    print("Encoding...")
    X = embed.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    preds = clf_priv.predict(X)

    true_conf = sum(1 for _, _, p in rows if p == "confidential")
    caught = sum(1 for (_, _, p), pred in zip(rows, preds) if p == "confidential" and pred == conf_idx)
    recall = caught / true_conf if true_conf else 0.0

    print("\n=== Pre-retrain baseline ===")
    print(f"IRR confidential recall: {caught}/{true_conf} = {recall*100:.1f}%")

    pred_dist = Counter(privacies[p] for (_, _, t), p in zip(rows, preds) if t == "confidential")
    print(f"Pred distribution for {true_conf} true-confidential IRR rows: {dict(pred_dist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
