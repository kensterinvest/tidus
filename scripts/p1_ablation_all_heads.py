#!/usr/bin/env python3
"""P1 ablation across ALL heads (domain, complexity, privacy).

No ground truth for domain/complexity on held-out — so we compare pool-level
distribution shifts + pairwise agreement. If P1 moved distributions meaningfully
on domain or complexity, that changes the "P1 did nothing" headline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = REPO_ROOT / "tests" / "classification" / "pool_chunks"
CHUNKS_DIR = REPO_ROOT / "tests" / "classification" / "chunks"
MAX_CHARS = 1200

CONDITIONS = [
    ("A pre",    REPO_ROOT / "tidus" / "classification" / "weights_b"),
    ("C ctrl",   REPO_ROOT / "tidus" / "classification" / "weights_b_ctrl"),
    ("B post",   REPO_ROOT / "tidus" / "classification" / "weights_b_p1"),
]

HEADS = [("domain", "domains"), ("complexity", "complexities"), ("privacy", "privacies")]


def main() -> int:
    pool_text: dict[str, str] = {}
    for pf in sorted(POOL_DIR.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                pool_text[r["id"]] = r["text"]

    labeled_ids = set()
    for lf in sorted(CHUNKS_DIR.glob("labels_*.jsonl")):
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                labeled_ids.add(json.loads(line)["id"])

    unlabeled = [(k, v) for k, v in pool_text.items() if k not in labeled_ids]
    print(f"Unlabeled pool: {len(unlabeled)}")
    texts = [t[:MAX_CHARS] for _, t in unlabeled]

    embed = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding...")
    X = embed.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    # per-head, per-condition predictions
    preds: dict[str, dict[str, list[int]]] = {h: {} for h, _ in HEADS}
    label_sets: dict[str, list[str]] = {}

    for label, wdir in CONDITIONS:
        mappings = json.loads((wdir / "label_mappings.json").read_text())
        for head_name, map_key in HEADS:
            clf = joblib.load(wdir / f"{head_name}_head.joblib")
            y = clf.predict(X)
            preds[head_name][label] = list(y)
            label_sets[head_name] = mappings[map_key]

    for head_name, _ in HEADS:
        print(f"\n=== Head: {head_name} ===")
        print(f"{'class':<16} " + " ".join(f"{c:>10}" for c, _ in CONDITIONS))
        classes = label_sets[head_name]
        n = len(texts)
        for cls in classes:
            row = [f"{cls:<16}"]
            for cond_label, _ in CONDITIONS:
                pp = preds[head_name][cond_label]
                count = sum(1 for p in pp if classes[p] == cls)
                row.append(f"{count/n*100:>9.2f}%")
            print(" ".join(row))

        print("Pairwise agreement:")
        labels = [c for c, _ in CONDITIONS]
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                a, b = preds[head_name][labels[i]], preds[head_name][labels[j]]
                agree = sum(1 for x, y in zip(a, b) if x == y)
                print(f"  {labels[i]:<6} vs {labels[j]:<6}  {agree}/{n} = {agree/n*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
