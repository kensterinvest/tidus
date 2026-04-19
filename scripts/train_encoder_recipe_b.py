#!/usr/bin/env python3
"""Phase 1 Recipe B — frozen sentence-transformer + class-weighted LR heads.

Alternative to Recipe A (LoRA-on-DeBERTa). Trains in seconds (no fine-tuning);
"safer at 3k scale" per plan.md. Used as a baseline when Recipe A underfits
or to triage class-imbalance failures.

Architecture:
    text -> all-MiniLM-L6-v2 (frozen, 384-dim) -> 3 sklearn LR heads
    Each LR uses class_weight="balanced" (natively handles 85/11/4 imbalance).

Usage:
    uv run python scripts/train_encoder_recipe_b.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_encoder import COMPLEXITIES, DOMAINS, PRIVACIES, SEED, load_joined_rows  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "tidus" / "classification" / "weights_b"
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 1200  # sentence-transformers caps at ~512 tokens; 1200 chars fits comfortably


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--C", type=float, default=1.0, help="LR inverse regularization")
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    rows = load_joined_rows()
    print(f"Loaded {len(rows)} rows")
    print(f"  Privacy: {dict(Counter(PRIVACIES[r.privacy] for r in rows))}")

    priv_labels = [r.privacy for r in rows]
    train_rows, val_rows = train_test_split(
        rows, test_size=0.15, random_state=SEED, stratify=priv_labels,
    )
    print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}")

    print(f"\nLoading embedding model: {args.embed_model}")
    embed = SentenceTransformer(args.embed_model)

    print("Encoding texts...")
    train_texts = [r.text[:MAX_CHARS] for r in train_rows]
    val_texts = [r.text[:MAX_CHARS] for r in val_rows]
    X_train = embed.encode(train_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    X_val = embed.encode(val_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    y_train_d = np.array([r.domain for r in train_rows])
    y_train_c = np.array([r.complexity for r in train_rows])
    y_train_p = np.array([r.privacy for r in train_rows])
    y_val_d = np.array([r.domain for r in val_rows])
    y_val_c = np.array([r.complexity for r in val_rows])
    y_val_p = np.array([r.privacy for r in val_rows])

    print("\nFitting 3 LR heads with class_weight='balanced'...")
    def fit(y):
        return LogisticRegression(
            class_weight="balanced", max_iter=args.max_iter,
            C=args.C, random_state=SEED, solver="lbfgs",
        ).fit(X_train, y)

    clf_domain = fit(y_train_d)
    clf_cmplx = fit(y_train_c)
    clf_priv = fit(y_train_p)

    print("\nVal metrics (full backtest in scripts/backtest_recipe_b.py):")
    for name, clf, y_val in [("Domain    ", clf_domain, y_val_d),
                              ("Complexity", clf_cmplx, y_val_c),
                              ("Privacy   ", clf_priv, y_val_p)]:
        y_pred = clf.predict(X_val)
        acc = (y_pred == y_val).mean()
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        print(f"  {name}: acc={acc:.3f}  macro-f1={f1:.3f}")

    y_priv_pred = clf_priv.predict(X_val)
    conf_idx = PRIVACIES.index("confidential")
    conf_mask = y_val_p == conf_idx
    if conf_mask.sum():
        tp = int(((y_priv_pred == conf_idx) & conf_mask).sum())
        gt = int(conf_mask.sum())
        print(f"  Confidential recall (gate): {tp}/{gt} = {tp/gt*100:.1f}%")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_domain, outdir / "domain_head.joblib")
    joblib.dump(clf_cmplx, outdir / "complexity_head.joblib")
    joblib.dump(clf_priv, outdir / "privacy_head.joblib")
    (outdir / "label_mappings.json").write_text(json.dumps({
        "domains": DOMAINS,
        "complexities": COMPLEXITIES,
        "privacies": PRIVACIES,
        "embed_model": args.embed_model,
        "max_chars": MAX_CHARS,
        "recipe": "B",
    }, indent=2))
    print(f"\nSaved to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
