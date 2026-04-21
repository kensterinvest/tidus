#!/usr/bin/env python3
"""P1 Step 5: Pre vs post encoder flag-rate on the 2,292 unlabeled pool.

Both encoders score the SAME set of prompts — the 2,292 pool prompts that
had no label before P1 sampling. We compare the resulting privacy-class
distributions. The overcaution-reduction story (109/200 flipped) should
manifest as fewer 'confidential' predictions on this macro pool.

Also reports agreement-rate between the two encoders.
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
PRE_WEIGHTS = REPO_ROOT / "tidus" / "classification" / "weights_b"
POST_WEIGHTS = REPO_ROOT / "tidus" / "classification" / "weights_b_p1"
MAX_CHARS = 1200


def main() -> int:
    # Load all labeled ids to exclude (2,653 + additional we added) -> unlabeled set is everything else
    labeled_ids: set[str] = set()
    for lf in sorted(CHUNKS_DIR.glob("labels_*.jsonl")):
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                labeled_ids.add(json.loads(line)["id"])
    print(f"Labeled ids: {len(labeled_ids)}")

    pool: list[tuple[str, str]] = []
    for pf in sorted(POOL_DIR.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["id"] not in labeled_ids:
                    pool.append((r["id"], r["text"]))
    print(f"Unlabeled pool (total): {len(pool)}")

    mappings_pre = json.loads((PRE_WEIGHTS / "label_mappings.json").read_text())
    mappings_post = json.loads((POST_WEIGHTS / "label_mappings.json").read_text())
    privacies = mappings_pre["privacies"]
    assert privacies == mappings_post["privacies"], "mapping mismatch"

    embed = SentenceTransformer(mappings_pre.get("embed_model", "all-MiniLM-L6-v2"))
    pre = joblib.load(PRE_WEIGHTS / "privacy_head.joblib")
    post = joblib.load(POST_WEIGHTS / "privacy_head.joblib")

    texts = [t[:MAX_CHARS] for _, t in pool]
    print("Encoding pool...")
    X = embed.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    pre_preds = pre.predict(X)
    post_preds = post.predict(X)

    pre_dist = Counter(privacies[p] for p in pre_preds)
    post_dist = Counter(privacies[p] for p in post_preds)
    n = len(pool)

    print(f"\n=== Pool flag-rate ({n} unlabeled prompts) ===")
    print(f"{'class':<15} {'pre':>10} {'post':>10} {'delta_pp':>10}")
    for cls in privacies:
        pre_p = pre_dist.get(cls, 0) / n * 100
        post_p = post_dist.get(cls, 0) / n * 100
        print(f"{cls:<15} {pre_p:>9.2f}%  {post_p:>9.2f}%  {post_p - pre_p:>+9.2f}")

    agree = sum(1 for a, b in zip(pre_preds, post_preds) if a == b)
    print(f"\nPre/post agreement: {agree}/{n} = {agree/n*100:.1f}%")

    flips: dict[tuple[str, str], int] = Counter()
    for a, b in zip(pre_preds, post_preds):
        if a != b:
            flips[(privacies[a], privacies[b])] += 1
    print(f"\nDisagreements ({sum(flips.values())} total):")
    for (a, b), cnt in sorted(flips.items(), key=lambda x: -x[1]):
        print(f"  {a:<12} -> {b:<12} {cnt:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
