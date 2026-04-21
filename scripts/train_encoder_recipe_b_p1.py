#!/usr/bin/env python3
"""P1 Step 2+3: Retrain Recipe B on the expanded corpus, with IRR ids held out.

Mirrors train_encoder_recipe_b.py exactly EXCEPT:
  * Rows whose id is in label_overrides_irr.jsonl are excluded BEFORE the split.
  * Weights are saved to weights_b_p1/ (the original weights_b/ stays intact).

The held-out IRR set is then scored by p1_step4_postbaseline.py for the clean
pre/post recall comparison (see advisor's 30-min protocol in session log).
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
from train_encoder import (  # noqa: E402
    CHUNKS_DIR,
    CMP2IDX,
    COMPLEXITIES,
    DOM2IDX,
    DOMAINS,
    POOL_DIR,
    PRIVACIES,
    PRV2IDX,
    REPO_ROOT,
    SEED,
    Row,
    _load_overrides,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_OUTPUT = REPO_ROOT / "tidus" / "classification" / "weights_b_p1"
IRR_PATH = REPO_ROOT / "tests" / "classification" / "label_overrides_irr.jsonl"
P1_LABEL_FILES = ["labels_066.jsonl", "labels_067.jsonl", "labels_068.jsonl", "labels_069.jsonl"]
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 1200


def load_irr_ids() -> set[str]:
    ids: set[str] = set()
    with IRR_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["id"])
    return ids


def load_p1_ids() -> set[str]:
    ids: set[str] = set()
    for fname in P1_LABEL_FILES:
        path = REPO_ROOT / "tests" / "classification" / "chunks" / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ids.add(json.loads(line)["id"])
    return ids


def load_joined_rows_with_ids(exclude_ids: set[str]) -> list[tuple[str, Row]]:
    """Mirror of train_encoder.load_joined_rows() but keeps ids AND excludes held-out set."""
    pool_text: dict[str, str] = {}
    for pf in sorted(POOL_DIR.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                pool_text[r["id"]] = r["text"]

    overrides = _load_overrides()
    rows: list[tuple[str, Row]] = []
    for lf in sorted(CHUNKS_DIR.glob("labels_*.jsonl")):
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["id"] not in pool_text or r["id"] in exclude_ids:
                    continue
                if r["id"] in overrides:
                    ov = overrides[r["id"]]
                    for k in ("domain", "complexity", "privacy"):
                        if k in ov:
                            r[k] = ov[k]
                rows.append((
                    r["id"],
                    Row(
                        text=pool_text[r["id"]],
                        domain=DOM2IDX[r["domain"]],
                        complexity=CMP2IDX[r["complexity"]],
                        privacy=PRV2IDX[r["privacy"]],
                    ),
                ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--exclude-p1", action="store_true", help="Also exclude P1 labels (control run)")
    parser.add_argument("--no-class-weight", action="store_true", help="Disable class_weight='balanced'")
    args = parser.parse_args()

    irr_ids = load_irr_ids()
    print(f"Held-out IRR ids: {len(irr_ids)}")

    exclude = set(irr_ids)
    if args.exclude_p1:
        p1_ids = load_p1_ids()
        print(f"Also excluding P1 ids: {len(p1_ids)}")
        exclude |= p1_ids
    print(f"Total exclusion set: {len(exclude)}")

    paired = load_joined_rows_with_ids(exclude)
    rows = [p[1] for p in paired]
    print(f"Training rows (IRR excluded): {len(rows)}")
    print(f"  Privacy: {dict(Counter(PRIVACIES[r.privacy] for r in rows))}")
    print(f"  Domain:  {dict(Counter(DOMAINS[r.domain] for r in rows))}")

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

    cw = None if args.no_class_weight else "balanced"
    print(f"\nFitting 3 LR heads with class_weight={cw!r}...")
    def fit(y):
        return LogisticRegression(
            class_weight=cw, max_iter=args.max_iter,
            C=args.C, random_state=SEED, solver="lbfgs",
        ).fit(X_train, y)

    clf_domain = fit(y_train_d)
    clf_cmplx = fit(y_train_c)
    clf_priv = fit(y_train_p)

    print("\nVal metrics:")
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
        print(f"  Val confidential recall: {tp}/{gt} = {tp/gt*100:.1f}%")

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
        "recipe": "B-p1-ctrl" if args.exclude_p1 else "B-p1",
        "excluded_ids_count": len(exclude),
        "excluded_irr": len(irr_ids),
        "excluded_p1": len(exclude) - len(irr_ids) if args.exclude_p1 else 0,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }, indent=2))
    print(f"\nSaved to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
