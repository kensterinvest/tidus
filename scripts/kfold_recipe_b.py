#!/usr/bin/env python3
"""5-fold stratified cross-validation gate check for Recipe B.

Problem it solves: the single train/val split yields only n=12 confidential
test cases, Wilson CI ±24pp — cannot resolve against the 95% gate.
K-fold pools out-of-fold predictions over all 1569 labels so every confidential
example is evaluated exactly once, n=79 confidentials, CI tightens to ±~5pp.

Each fold: encode texts with all-MiniLM-L6-v2 (frozen), fit three
LogisticRegression(class_weight="balanced") heads on 4/5 of the data, predict
on the held-out fifth. Aggregate predictions across folds, then compute the
same gate metrics as scripts/backtest_recipe_b.py.

Usage:
    uv run python scripts/kfold_recipe_b.py
    uv run python scripts/kfold_recipe_b.py --folds 5 --embed-model all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    rows = load_joined_rows()
    print(f"Loaded {len(rows)} rows  ({dict(Counter(PRIVACIES[r.privacy] for r in rows))})")

    y_d = np.array([r.domain for r in rows])
    y_c = np.array([r.complexity for r in rows])
    y_p = np.array([r.privacy for r in rows])

    print(f"\nEncoding {len(rows)} texts with {args.embed_model} (one-time)...")
    embed = SentenceTransformer(args.embed_model)
    X = embed.encode(
        [r.text[:MAX_CHARS] for r in rows],
        batch_size=32, show_progress_bar=True, normalize_embeddings=True,
    )

    def make_lr():
        return LogisticRegression(
            class_weight="balanced", max_iter=args.max_iter,
            C=args.C, random_state=SEED, solver="lbfgs",
        )

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    d_pred = np.full(len(rows), -1, dtype=int)
    c_pred = np.full(len(rows), -1, dtype=int)
    p_pred = np.full(len(rows), -1, dtype=int)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y_p), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        print(f"\nFold {fold}/{args.folds}  (train={len(tr_idx)}, test={len(te_idx)})")
        d_pred[te_idx] = make_lr().fit(X_tr, y_d[tr_idx]).predict(X_te)
        c_pred[te_idx] = make_lr().fit(X_tr, y_c[tr_idx]).predict(X_te)
        p_pred[te_idx] = make_lr().fit(X_tr, y_p[tr_idx]).predict(X_te)

    assert (d_pred != -1).all() and (c_pred != -1).all() and (p_pred != -1).all()

    n = len(rows)
    dom_ok = int((d_pred == y_d).sum())
    cmp_ok = int((c_pred == y_c).sum())
    conf_mask = y_p == CONF_IDX
    conf_ground = int(conf_mask.sum())
    conf_flagged = int(((p_pred == CONF_IDX) & conf_mask).sum())
    conf_predicted = int((p_pred == CONF_IDX).sum())

    sep = "=" * 70
    print(f"\n{sep}\n  Recipe B — {args.folds}-fold cross-validation  (n={n})\n{sep}")

    dp, dlo, dhi = wilson_ci(dom_ok, n)
    print(f"\n  Domain accuracy:       {dom_ok}/{n} = {dp*100:.2f}%   CI [{dlo*100:.2f}, {dhi*100:.2f}]")
    print(f"    Gate >= 85%  -> {gate_verdict(dlo, dhi, 0.85)}")

    cp, clo, chi = wilson_ci(cmp_ok, n)
    print(f"\n  Complexity accuracy:   {cmp_ok}/{n} = {cp*100:.2f}%   CI [{clo*100:.2f}, {chi*100:.2f}]")

    pp, plo, phi = wilson_ci(conf_flagged, conf_ground)
    print(f"\n  Confidential recall:   {conf_flagged}/{conf_ground} = {pp*100:.2f}%   CI [{plo*100:.2f}, {phi*100:.2f}]")
    print(f"    Gate >= 95%  -> {gate_verdict(plo, phi, 0.95)}")

    if conf_predicted:
        print(f"  Confidential precision: {conf_flagged}/{conf_predicted} = {conf_flagged/conf_predicted*100:.2f}%")

    print("\n  Privacy confusion (rows=actual, cols=predicted)")
    print("    " + " ".join(f"{v[:7]:>7}" for v in PRIVACIES) + "   | total")
    for ai, aname in enumerate(PRIVACIES):
        mask = y_p == ai
        tot = int(mask.sum())
        row = f"    {aname[:13]:>13}"
        for pi in range(len(PRIVACIES)):
            c = int(((p_pred == pi) & mask).sum())
            row += f" {c:>6d} " if c else "      . "
        print(row + f"  | {tot}")

    dom_pass = dlo >= 0.85
    conf_pass = plo >= 0.95
    print("\n" + sep)
    print("  PHASE 0 GATE (Recipe B, k-fold):")
    if dom_pass and conf_pass:
        print("  -> PASS: Recipe B may ship to Tier 2.")
        return 0
    if dhi < 0.85 or phi < 0.95:
        print("  -> FAIL: at least one axis CI upper-bound below gate.")
        return 1
    print("  -> INCONCLUSIVE (CI straddles at least one gate).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
