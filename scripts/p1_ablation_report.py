#!/usr/bin/env python3
"""P1 ablation: compare pre (A), control (C, -IRR -P1), post (B, -IRR +P1).

A vs C isolates IRR-exclusion effect.
C vs B isolates P1-addition effect.
Reports IRR recall and pool flag-rate for all three.
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
CHUNKS_DIR = REPO_ROOT / "tests" / "classification" / "chunks"
IRR_PATH = REPO_ROOT / "tests" / "classification" / "label_overrides_irr.jsonl"
MAX_CHARS = 1200

CONDITIONS = [
    ("A (pre: +IRR +P1, leaked)", REPO_ROOT / "tidus" / "classification" / "weights_b"),
    ("C (control: -IRR -P1)",      REPO_ROOT / "tidus" / "classification" / "weights_b_ctrl"),
    ("B (post: -IRR +P1)",          REPO_ROOT / "tidus" / "classification" / "weights_b_p1"),
]


def build_irr_rows(pool_text: dict) -> list[tuple[str, str, str]]:
    irr = []
    with IRR_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            irr.append(json.loads(line))
    rows = [(r["id"], pool_text[r["id"]], r["privacy"]) for r in irr if r["id"] in pool_text]
    return rows


def build_unlabeled_pool(pool_text: dict) -> list[tuple[str, str]]:
    labeled_ids = set()
    for lf in sorted(CHUNKS_DIR.glob("labels_*.jsonl")):
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                labeled_ids.add(json.loads(line)["id"])
    return [(k, v) for k, v in pool_text.items() if k not in labeled_ids]


def main() -> int:
    pool_text: dict[str, str] = {}
    for pf in sorted(POOL_DIR.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                pool_text[r["id"]] = r["text"]

    irr_rows = build_irr_rows(pool_text)
    unlabeled = build_unlabeled_pool(pool_text)
    print(f"IRR rows (joined): {len(irr_rows)}")
    print(f"Unlabeled pool:    {len(unlabeled)}")

    embed = SentenceTransformer("all-MiniLM-L6-v2")

    irr_texts = [t[:MAX_CHARS] for _, t, _ in irr_rows]
    pool_texts = [t[:MAX_CHARS] for _, t in unlabeled]
    print("Encoding IRR + pool (one-shot)...")
    X_irr = embed.encode(irr_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    X_pool = embed.encode(pool_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    print("\n" + "=" * 78)
    print(f"{'Condition':<32} {'IRR recall':<14} {'pub%':>8} {'int%':>8} {'conf%':>8}")
    print("=" * 78)

    pool_preds: dict[str, list[int]] = {}
    for label, wdir in CONDITIONS:
        mappings = json.loads((wdir / "label_mappings.json").read_text())
        privacies = mappings["privacies"]
        conf_idx = privacies.index("confidential")
        clf = joblib.load(wdir / "privacy_head.joblib")

        irr_preds = clf.predict(X_irr)
        true_conf = sum(1 for _, _, p in irr_rows if p == "confidential")
        caught = sum(1 for (_, _, p), pp in zip(irr_rows, irr_preds) if p == "confidential" and pp == conf_idx)
        recall = f"{caught}/{true_conf} = {caught/true_conf*100:.1f}%"

        pp = clf.predict(X_pool)
        pool_preds[label] = list(pp)
        dist = Counter(privacies[i] for i in pp)
        n = len(pp)
        pub = dist.get("public", 0) / n * 100
        inl = dist.get("internal", 0) / n * 100
        conf = dist.get("confidential", 0) / n * 100
        print(f"{label:<32} {recall:<14} {pub:>7.2f}% {inl:>7.2f}% {conf:>7.2f}%")

    # Pairwise agreement
    print("\n" + "=" * 78)
    print("Pairwise pool agreement:")
    labels = list(pool_preds.keys())
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            a, b = pool_preds[labels[i]], pool_preds[labels[j]]
            agree = sum(1 for x, y in zip(a, b) if x == y)
            print(f"  {labels[i]:<32} vs {labels[j]:<32} {agree}/{len(a)} = {agree/len(a)*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
