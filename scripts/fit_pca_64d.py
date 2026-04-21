#!/usr/bin/env python3
"""Fit a 384->64 PCA projection for Stage B PII-safe telemetry.

Why: plan.md §Stage B requires emitting `embedding_reduced_64d` per request.
Raw 384-d all-MiniLM embeddings are semi-reversible (nearest-neighbour lookup
against a public corpus recovers the prompt). Reducing to 64-d via PCA keeps
enough semantic signal for similarity search in offline analysis while
making inversion meaningfully harder.

Method:
    1. Load all labeled prompts from tests/classification/chunks/labels_*.jsonl
       (IDs → pool texts). This is the in-distribution corpus for Tidus traffic.
    2. Encode with all-MiniLM-L6-v2 (same model the runtime uses).
    3. Fit sklearn PCA(n_components=64) on the encoded matrix.
    4. Save to tidus/classification/weights_b/pca_64d.joblib.

Usage:
    uv run python scripts/fit_pca_64d.py
    uv run python scripts/fit_pca_64d.py --output custom/path/pca.joblib

Re-run this when:
    * A new labeled chunk lands (labels_NNN.jsonl) that materially shifts the
      distribution (e.g., first internal dataset added).
    * The encoder backbone changes (e.g., swap MiniLM for BGE-small).
The artifact is stable across minor WildChat additions — no need to refit per
session.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = REPO_ROOT / "tests" / "classification" / "chunks"
POOL_DIR = REPO_ROOT / "tests" / "classification" / "pool_chunks"
WEIGHTS_DIR = REPO_ROOT / "tidus" / "classification" / "weights_b"

EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 1200  # matches runtime encoder + train_encoder_recipe_b.py
N_COMPONENTS = 64


def _load_labeled_ids() -> set[str]:
    ids: set[str] = set()
    for f in sorted(CHUNKS_DIR.glob("labels_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def _load_pool() -> dict[str, str]:
    pool: dict[str, str] = {}
    for f in sorted(POOL_DIR.glob("pool_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                pool[row["id"]] = row["text"]
    return pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(WEIGHTS_DIR / "pca_64d.joblib"),
        help="Path for the fitted PCA joblib artifact",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoding batch size (raise if VRAM allows)",
    )
    args = parser.parse_args()

    labeled_ids = _load_labeled_ids()
    pool = _load_pool()
    texts = [pool[i] for i in labeled_ids if i in pool]
    if not texts:
        print("ERROR: no labeled-prompt texts found.", file=sys.stderr)
        return 1
    print(f"Fitting PCA on {len(texts)} labeled prompts.")

    print(f"Encoding with {EMBED_MODEL} (cap {MAX_CHARS} chars)...")
    embed = SentenceTransformer(EMBED_MODEL)
    X = embed.encode(
        [t[:MAX_CHARS] for t in texts],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    X = np.asarray(X, dtype=np.float32)
    print(f"Embedding matrix shape: {X.shape}")

    if X.shape[0] <= N_COMPONENTS:
        print(
            f"ERROR: n_samples={X.shape[0]} must exceed n_components={N_COMPONENTS}. "
            "Label more prompts or lower N_COMPONENTS.",
            file=sys.stderr,
        )
        return 1

    print(f"Fitting PCA(n_components={N_COMPONENTS})...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(X)
    print(
        f"Explained variance: sum={pca.explained_variance_ratio_.sum():.4f}  "
        f"(first 10 components: {[f'{v:.3f}' for v in pca.explained_variance_ratio_[:10]]})",
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
