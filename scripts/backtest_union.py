#!/usr/bin/env python3
"""Gate check on UNION(POC regex + Recipe B encoder).

The production architecture's privacy rule (boot / plan.md) is:
    ANY hit across regex / keywords / Presidio / encoder -> confidential

So evaluating the encoder ALONE against the 95% gate underestimates what the
deployed system would achieve. This script computes the empirical union of
POC Tier 1 + Recipe B encoder (k-fold out-of-fold predictions), skipping
Tier 2b Presidio for now (it would only add recall on top — an upper bound).

Two runs are compared:
    - pre-audit labels (original)    -- not reachable here; see original backtest
    - post-audit labels (overrides applied via train_encoder.load_joined_rows)

Usage:
    uv run python scripts/backtest_union.py
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poc_classifier import Privacy as POCPrivacy
from poc_classifier import classify_t1  # noqa: E402
from train_encoder import PRIVACIES, PRV2IDX, SEED, load_joined_rows  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONF_IDX = PRV2IDX["confidential"]
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 1200


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - halfw), min(1.0, center + halfw)


def gate_verdict(lo: float, hi: float, threshold: float) -> str:
    if lo >= threshold:
        return f"PASS (CI lower {lo*100:.1f}% >= {threshold*100:.0f}%)"
    if hi < threshold:
        return f"FAIL (CI upper {hi*100:.1f}% < {threshold*100:.0f}%)"
    return f"INCONCLUSIVE (CI [{lo*100:.1f}, {hi*100:.1f}] straddles {threshold*100:.0f}%)"


def main() -> int:
    rows = load_joined_rows()
    y_p = np.array([r.privacy for r in rows])
    texts = [r.text for r in rows]
    n = len(rows)
    print(f"Loaded {n} rows  (privacy: {dict(Counter(PRIVACIES[p] for p in y_p))})")

    # --- POC Tier 1 ---
    print("\nRunning POC Tier 1 classifier...")
    poc_conf = np.zeros(n, dtype=bool)
    for i, t in enumerate(texts):
        res = classify_t1(t)
        poc_conf[i] = (res.privacy == POCPrivacy.confidential)
        if (i + 1) % 400 == 0:
            print(f"  ...{i+1}/{n}")

    # --- Recipe B k-fold ---
    print(f"\nEncoding {n} texts with {EMBED_MODEL}...")
    embed = SentenceTransformer(EMBED_MODEL)
    X = embed.encode([t[:MAX_CHARS] for t in texts],
                     batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    print("Recipe B 5-fold out-of-fold predictions...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    enc_pred = np.full(n, -1, dtype=int)
    for fold, (tr, te) in enumerate(skf.split(X, y_p), 1):
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            random_state=SEED, solver="lbfgs",
        ).fit(X[tr], y_p[tr])
        enc_pred[te] = clf.predict(X[te])
        print(f"  fold {fold}/5 done")
    enc_conf = enc_pred == CONF_IDX

    # --- Union + diagnostics ---
    union_conf = poc_conf | enc_conf
    intersect_conf = poc_conf & enc_conf
    conf_mask = y_p == CONF_IDX
    gt_conf = int(conf_mask.sum())

    def report(name: str, flag: np.ndarray):
        tp = int((flag & conf_mask).sum())
        fp = int((flag & ~conf_mask).sum())
        predicted = int(flag.sum())
        pp, plo, phi = wilson_ci(tp, gt_conf)
        print(f"\n  {name}")
        print(f"    Confidential recall: {tp}/{gt_conf} = {pp*100:.2f}%   CI [{plo*100:.2f}, {phi*100:.2f}]")
        print(f"    Gate >= 95%:         {gate_verdict(plo, phi, 0.95)}")
        print(f"    Total flagged: {predicted} (TP={tp}, FP={fp}, precision={tp/predicted*100:.1f}% if predicted)" if predicted else "    Total flagged: 0")

    sep = "=" * 72
    print(f"\n{sep}\n  UNION(POC + Recipe B encoder) — privacy gate check  (n={n}, gt_conf={gt_conf})\n{sep}")

    report("POC Tier 1 alone (regex/PII heuristics)", poc_conf)
    report("Recipe B encoder alone (k-fold)", enc_conf)
    report("UNION (either fires)", union_conf)
    report("INTERSECTION (both fire — high-precision subset)", intersect_conf)

    # --- Overlap analysis on the ground-truth confidentials ---
    both = int(((poc_conf & enc_conf) & conf_mask).sum())
    only_poc = int(((poc_conf & ~enc_conf) & conf_mask).sum())
    only_enc = int(((~poc_conf & enc_conf) & conf_mask).sum())
    neither = int(((~poc_conf & ~enc_conf) & conf_mask).sum())
    print(f"\n  Breakdown of {gt_conf} ground-truth confidentials:")
    print(f"    caught by BOTH:        {both}")
    print(f"    caught by POC only:    {only_poc}")
    print(f"    caught by encoder only:{only_enc}")
    print(f"    caught by NEITHER:     {neither}")
    print(f"    (union should = both + only_poc + only_enc = {both + only_poc + only_enc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
