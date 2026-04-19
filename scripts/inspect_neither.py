#!/usr/bin/env python3
"""Inspect ground-truth confidentials that NEITHER POC nor Recipe B encoder caught.

These are the hard core of the gate failure. The advisor's framework: eyeball
and categorize each into one of three buckets:
    (a) residual labeler overcall (flip to public/internal and rerun audit)
    (b) genuine hard PII the regex missed (Presidio candidate)
    (c) genuinely ambiguous / taxonomy underspecified

Writes the 18 to a text report with (id, chunk, rationale, first-200-chars).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poc_classifier import Privacy as POCPrivacy, classify_t1  # noqa: E402
from train_encoder import PRIVACIES, PRV2IDX, SEED, load_joined_rows  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
CHUNKS = REPO / "tests" / "classification" / "chunks"
POOL = REPO / "tests" / "classification" / "pool_chunks"
CONF_IDX = PRV2IDX["confidential"]
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 1200


def main() -> int:
    rows = load_joined_rows()
    texts = [r.text for r in rows]
    y_p = np.array([r.privacy for r in rows])
    n = len(rows)

    # Need the id and source chunk / rationale per row -- re-read labels
    pool_text = {}
    for pf in sorted(POOL.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                pool_text[r["id"]] = r["text"]
    id_list: list[tuple[str, str, str]] = []  # (id, chunk, rationale)
    for lf in sorted(CHUNKS.glob("labels_*.jsonl")):
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["id"] in pool_text:
                    id_list.append((r["id"], lf.stem, r.get("rationale", "")))
    assert len(id_list) == n

    print(f"Loaded {n} rows — running POC + k-fold Recipe B...")
    # POC
    poc_conf = np.zeros(n, dtype=bool)
    for i, t in enumerate(texts):
        poc_conf[i] = (classify_t1(t).privacy == POCPrivacy.confidential)

    # Recipe B 5-fold OOF
    embed = SentenceTransformer(EMBED_MODEL)
    X = embed.encode([t[:MAX_CHARS] for t in texts],
                     batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    enc_pred = np.full(n, -1, dtype=int)
    for tr, te in skf.split(X, y_p):
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            random_state=SEED, solver="lbfgs",
        ).fit(X[tr], y_p[tr])
        enc_pred[te] = clf.predict(X[te])
    enc_conf = enc_pred == CONF_IDX

    # NEITHER-caught ground-truth confidentials
    conf_mask = y_p == CONF_IDX
    neither = conf_mask & ~poc_conf & ~enc_conf
    idxs = np.where(neither)[0]

    report = []
    report.append(f"# NEITHER-caught confidentials ({len(idxs)} of {int(conf_mask.sum())} ground-truth)")
    report.append("")
    report.append("For each, write flip decision BEFORE looking at gate impact:")
    report.append("  KEEP   = genuinely confidential, encoder/POC just missed it")
    report.append("  FLIP   = was labeler overcall, should not be confidential")
    report.append("  AMBIG  = taxonomy underspecified, no clear answer")
    report.append("")
    report.append("=" * 80)
    for idx in idxs:
        cid, chunk, rationale = id_list[idx]
        text = texts[idx]
        preview = " ".join(text.split())[:400]
        report.append("")
        report.append(f"[{chunk}] {cid}")
        report.append(f"  rationale: {rationale[:240]}")
        report.append(f"  text_len:  {len(text)}")
        report.append(f"  text:      {preview}")

    out_path = REPO / "audit_neither.txt"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote {out_path}  ({len(idxs)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
