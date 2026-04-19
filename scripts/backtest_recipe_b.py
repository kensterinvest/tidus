#!/usr/bin/env python3
"""Backtest Recipe B (frozen sentence-transformer + LR) against the Phase 0 gate.

Mirrors scripts/backtest_trained_encoder.py but for sklearn LR heads on
frozen sentence-transformer embeddings.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_encoder import PRIVACIES, PRV2IDX, SEED, Row, load_joined_rows  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "tidus" / "classification" / "weights_b"
CONF_IDX = PRV2IDX["confidential"]


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


def report_slice(name: str, rows: list[Row],
                 d_pred: np.ndarray, c_pred: np.ndarray, p_pred: np.ndarray,
                 apply_gate: bool) -> tuple[bool, bool]:
    n = len(rows)
    d_true = np.array([r.domain for r in rows])
    c_true = np.array([r.complexity for r in rows])
    p_true = np.array([r.privacy for r in rows])

    dom_ok = int((d_pred == d_true).sum())
    cmp_ok = int((c_pred == c_true).sum())
    conf_mask = p_true == CONF_IDX
    conf_ground = int(conf_mask.sum())
    conf_flagged = int(((p_pred == CONF_IDX) & conf_mask).sum())
    conf_predicted = int((p_pred == CONF_IDX).sum())

    sep = "=" * 70
    print(f"\n{sep}\n  {name}  (n={n})\n{sep}")

    dp, dlo, dhi = wilson_ci(dom_ok, n)
    print(f"\n  Domain accuracy: {dom_ok}/{n} = {dp*100:.2f}%   CI [{dlo*100:.2f}, {dhi*100:.2f}]")
    if apply_gate:
        print(f"    Gate >= 85%  -> {gate_verdict(dlo, dhi, 0.85)}")

    cp, clo, chi = wilson_ci(cmp_ok, n)
    print(f"\n  Complexity accuracy: {cmp_ok}/{n} = {cp*100:.2f}%   CI [{clo*100:.2f}, {chi*100:.2f}]")

    dom_pass = dlo >= 0.85
    dom_fail = dhi < 0.85

    if conf_ground:
        pp, plo, phi = wilson_ci(conf_flagged, conf_ground)
        print(f"\n  Confidential recall: {conf_flagged}/{conf_ground} = {pp*100:.2f}%   CI [{plo*100:.2f}, {phi*100:.2f}]")
        if apply_gate:
            print(f"    Gate >= 95%  -> {gate_verdict(plo, phi, 0.95)}")
        conf_pass = plo >= 0.95
        conf_fail = phi < 0.95
    else:
        print("\n  Confidential recall: n/a")
        conf_pass = False
        conf_fail = False

    if conf_predicted:
        print(f"  Confidential precision: {conf_flagged}/{conf_predicted} = {conf_flagged/conf_predicted*100:.2f}%")

    print("\n  Privacy confusion (rows=actual, cols=predicted)")
    print("    " + " ".join(f"{v[:7]:>7}" for v in PRIVACIES) + "   | total")
    for ai, aname in enumerate(PRIVACIES):
        mask = p_true == ai
        tot = int(mask.sum())
        row = f"    {aname[:13]:>13}"
        for pi in range(len(PRIVACIES)):
            c = int(((p_pred == pi) & mask).sum())
            row += f" {c:>6d} " if c else "      . "
        print(row + f"  | {tot}")

    if apply_gate:
        print(f"\n  Gate decision: Domain {'PASS' if dom_pass else ('FAIL' if dom_fail else 'INCONCLUSIVE')}  |  "
              f"Privacy {'PASS' if conf_pass else ('FAIL' if conf_fail else 'INCONCLUSIVE')}")
    return (dom_pass and conf_pass, dom_fail or conf_fail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    if not (weights_dir / "label_mappings.json").exists():
        print(f"ERROR: no Recipe B weights at {weights_dir}.")
        print("Run `uv run python scripts/train_encoder_recipe_b.py` first.")
        return 1

    mappings = json.loads((weights_dir / "label_mappings.json").read_text())
    print(f"Loading Recipe B from {weights_dir}")
    print(f"  Embedding model: {mappings['embed_model']}")
    max_chars = mappings.get("max_chars", 1200)

    clf_domain = joblib.load(weights_dir / "domain_head.joblib")
    clf_cmplx = joblib.load(weights_dir / "complexity_head.joblib")
    clf_priv = joblib.load(weights_dir / "privacy_head.joblib")
    embed = SentenceTransformer(mappings["embed_model"])

    rows = load_joined_rows()
    print(f"Joined {len(rows)} label rows  ({dict(Counter(PRIVACIES[r.privacy] for r in rows))})")

    priv_labels = [r.privacy for r in rows]
    train_rows, val_rows = train_test_split(
        rows, test_size=0.15, random_state=SEED, stratify=priv_labels,
    )
    print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}")

    def infer(slice_rows):
        texts = [r.text[:max_chars] for r in slice_rows]
        X = embed.encode(texts, batch_size=args.batch_size, show_progress_bar=False, normalize_embeddings=True)
        return clf_domain.predict(X), clf_cmplx.predict(X), clf_priv.predict(X)

    print("\nInference on val set...")
    d_v, c_v, p_v = infer(val_rows)
    print("Inference on full set (diagnostic)...")
    d_a, c_a, p_a = infer(rows)

    val_pass, val_fail = report_slice("VAL SLICE (gate decision)", val_rows, d_v, c_v, p_v, apply_gate=True)
    report_slice("FULL SET (diagnostic — includes train)", rows, d_a, c_a, p_a, apply_gate=False)

    print("\n" + "=" * 70)
    print("  PHASE 0 GATE (Recipe B, val-slice):")
    if val_pass:
        print("  -> PASS: Recipe B may ship to Tier 2.")
        return 0
    if val_fail:
        print("  -> FAIL: neither recipe clears gate on current labels — audit label quality or get more data.")
        return 1
    print("  -> INCONCLUSIVE: val CI straddles gate — need more labels or k-fold.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
