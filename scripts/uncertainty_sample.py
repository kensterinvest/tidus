#!/usr/bin/env python3
"""Lever P1 — uncertainty-sampled active learning on the unlabeled WildChat pool.

Per plan.md "Pre-Adoption Research Programme" Lever P1.

Method (Settles 2009, most-cited active-learning approach):
    1. Load trained Recipe B encoder (frozen sentence-transformer + 3 sklearn LR heads).
    2. Build the unlabeled set (pool prompts not yet in chunks/labels_*.jsonl).
    3. Encode unlabeled prompts via all-MiniLM-L6-v2 (same preprocessing as Recipe B).
    4. Score with the privacy head -> per-class softmax probabilities.
    5. Compute uncertainty = 1 - (p(top-1) - p(top-2))  (lower margin = higher uncertainty).
    6. Stratify by hard-argmax predicted class (public / internal / confidential).
    7. Pick the N most-uncertain in each stratum (default: 60 public / 70 internal / 70 confidential = 200).
       If a stratum is smaller than its target, take all of it and fill the remainder
       from the most populous stratum.
    8. Emit a labeling pack with one JSON per line, including the raw text and the
       encoder's current prediction + probabilities (as labeling hints, not ground truth).

Why privacy head only (cycle 1): privacy is the axis the 89.2% baseline is measured on
and the axis with the worst inter-rater agreement (Fleiss kappa = 0.577). Sampling
for domain/complexity uncertainty would pick different prompts and dilute the signal
for the metric that matters. Extend to other axes in cycle 2 if P1 succeeds.

Usage:
    uv run python scripts/uncertainty_sample.py
    uv run python scripts/uncertainty_sample.py --n 200 --target-confidential 70 --target-internal 70 --target-public 60
    uv run python scripts/uncertainty_sample.py --output tests/classification/p1_uncertain/pack_001.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "tidus" / "classification" / "weights_b"
POOL_DIR = REPO_ROOT / "tests" / "classification" / "pool_chunks"
CHUNKS_DIR = REPO_ROOT / "tests" / "classification" / "chunks"
MASTER_POOL = REPO_ROOT / "tests" / "classification" / "prompts_pool.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "classification" / "p1_uncertain" / "pack_001.jsonl"

EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 1200  # matches train_encoder_recipe_b.py


def load_labeled_ids() -> set[str]:
    """All prompt IDs that have already been labeled (and so should be excluded)."""
    ids: set[str] = set()
    for labels_file in sorted(CHUNKS_DIR.glob("labels_*.jsonl")):
        for line in labels_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def load_pool() -> dict[str, str]:
    """Prompt ID -> text, preferring per-chunk files (what train_encoder_recipe_b uses).

    Falls back to the master prompts_pool.jsonl if pool_chunks/ is empty.
    """
    pool: dict[str, str] = {}
    chunks = sorted(POOL_DIR.glob("pool_*.jsonl"))
    if chunks:
        for pf in chunks:
            for line in pf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    pool[row["id"]] = row["text"]
        return pool
    # Fallback
    for line in MASTER_POOL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pool[row["id"]] = row["text"]
    return pool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Total number of prompts to sample")
    parser.add_argument("--target-public", type=int, default=60)
    parser.add_argument("--target-internal", type=int, default=70)
    parser.add_argument("--target-confidential", type=int, default=70)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    targets = {
        "public": args.target_public,
        "internal": args.target_internal,
        "confidential": args.target_confidential,
    }
    assert sum(targets.values()) == args.n, (
        f"stratum targets {targets} must sum to --n={args.n}"
    )

    # --- 1. Load encoder heads + label mappings ---
    mappings = json.loads((WEIGHTS_DIR / "label_mappings.json").read_text(encoding="utf-8"))
    privacy_labels = mappings["privacies"]  # ["public", "internal", "confidential"]
    privacy_head = joblib.load(WEIGHTS_DIR / "privacy_head.joblib")

    # sklearn LR stores classes_ as integers mapped to our privacy labels
    lr_classes = list(privacy_head.classes_)
    # Verify ordering: lr_classes[i] is the integer label; privacy_labels[lr_classes[i]] is the string
    print(f"Encoder loaded. Privacy head classes (int -> str): "
          f"{[(i, privacy_labels[i]) for i in lr_classes]}")

    # --- 2. Identify unlabeled prompts ---
    labeled_ids = load_labeled_ids()
    pool = load_pool()
    unlabeled_ids = [pid for pid in pool if pid not in labeled_ids]
    print(f"Labeled: {len(labeled_ids)}  Pool: {len(pool)}  Unlabeled: {len(unlabeled_ids)}")

    if len(unlabeled_ids) < args.n:
        print(f"WARNING: only {len(unlabeled_ids)} unlabeled prompts; requested {args.n}.",
              file=sys.stderr)

    # --- 3. Encode ---
    print(f"\nEncoding {len(unlabeled_ids)} prompts with {EMBED_MODEL}...")
    embed = SentenceTransformer(EMBED_MODEL)
    texts = [pool[pid][:MAX_CHARS] for pid in unlabeled_ids]
    X = embed.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # --- 4. Privacy-head probabilities ---
    print("Scoring privacy head...")
    probs = privacy_head.predict_proba(X)  # shape (N, 3) in lr_classes order

    # Re-index columns to privacy_labels order (public, internal, confidential)
    # so probs[:, 0] = P(public), probs[:, 1] = P(internal), probs[:, 2] = P(confidential)
    col_order = [lr_classes.index(i) for i in range(len(privacy_labels))]
    probs = probs[:, col_order]

    # --- 5. Uncertainty = 1 - margin ---
    sorted_probs = np.sort(probs, axis=1)  # ascending: [p3, p2, p1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]  # p(top1) - p(top2)
    uncertainty = 1.0 - margin
    hard_pred = probs.argmax(axis=1)  # 0/1/2

    # --- 6. Stratify + 7. Pick top-N per stratum ---
    print(f"\nHard-prediction distribution on unlabeled pool: "
          f"{dict(Counter(privacy_labels[p] for p in hard_pred))}")

    picked: list[int] = []
    leftover: dict[str, int] = {}
    by_class_all: dict[str, np.ndarray] = {}
    for cls_idx, cls_name in enumerate(privacy_labels):
        cls_mask = hard_pred == cls_idx
        cls_indices = np.where(cls_mask)[0]
        # sort by uncertainty DESC (highest uncertainty first)
        cls_sorted = cls_indices[np.argsort(-uncertainty[cls_indices])]
        by_class_all[cls_name] = cls_sorted
        want = targets[cls_name]
        take = min(want, len(cls_sorted))
        picked.extend(cls_sorted[:take].tolist())
        leftover[cls_name] = want - take
        print(f"  {cls_name}: target={want} available={len(cls_sorted)} "
              f"taken={take} deficit={leftover[cls_name]}")

    # Backfill any deficit from the most populous remaining-class pool
    total_deficit = sum(leftover.values())
    if total_deficit > 0:
        picked_set = set(picked)
        # Rank all remaining unpicked prompts globally by uncertainty
        remaining = np.array([i for i in range(len(unlabeled_ids)) if i not in picked_set])
        remaining_sorted = remaining[np.argsort(-uncertainty[remaining])]
        fill = remaining_sorted[:total_deficit].tolist()
        picked.extend(fill)
        print(f"  Backfilled {len(fill)} prompts from the global remainder to cover deficits.")

    picked = picked[: args.n]
    print(f"\nFinal pack size: {len(picked)} (target {args.n})")

    # --- 8. Emit pack ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx in picked:
            pid = unlabeled_ids[idx]
            record = {
                "id": pid,
                "text": pool[pid],
                "encoder_predicted_privacy": privacy_labels[hard_pred[idx]],
                "encoder_margin": round(float(margin[idx]), 4),
                "encoder_probs": {
                    privacy_labels[i]: round(float(probs[idx, i]), 4)
                    for i in range(len(privacy_labels))
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(picked)} prompts to {output_path}")

    # Summary stats
    picked_preds = [privacy_labels[hard_pred[i]] for i in picked]
    picked_margins = [margin[i] for i in picked]
    print("\nPack composition (by encoder's current hard prediction):")
    for cls, n in sorted(Counter(picked_preds).items()):
        print(f"  {cls}: {n}")
    print(f"\nMargin distribution (lower = more uncertain):")
    print(f"  min={min(picked_margins):.4f}  "
          f"median={float(np.median(picked_margins)):.4f}  "
          f"max={max(picked_margins):.4f}")

    print("\nNext steps:")
    print("  1. Label the emitted pack (see plan.md 'Pre-Adoption Research Programme' - use the")
    print("     same SYSTEM_PROMPT rubric as scripts/label_wildchat.py, unchanged).")
    print("  2. Save labels to tests/classification/chunks/labels_066.jsonl (next sequential).")
    print("  3. Re-run: uv run python scripts/train_encoder_recipe_b.py")
    print("  4. Measure: uv run python scripts/backtest_recipe_b.py")
    print("  5. Compare pre- vs. post-retrain recall on the 83-confidential IRR ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
