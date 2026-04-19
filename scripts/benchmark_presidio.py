#!/usr/bin/env python3
"""Presidio benchmark — recall contribution to UNION + per-request latency.

Per plan.md Phase 0.5 benchmark:
    p95 <= 30ms -> keep as parallel Tier 2b
    p95 >  30ms -> demote to conditional Tier 3 (trigger: low encoder conf + no Tier 1 hit)

Runs Presidio AnalyzerEngine on all 1569 joined rows and reports:
  - Presidio-alone confidential recall (treating ANY PII hit as `confidential`)
  - UNION(POC + Recipe B encoder + Presidio) recall — the full architecture gate
  - per-request latency (p50, p95, p99)

Deps installed 2026-04-19: presidio-analyzer==2.2.362, spacy==3.8.14, en_core_web_sm==3.8.0
(Python pinned to 3.13 — spaCy has no 3.14 wheel.)

Usage:
    uv run python scripts/benchmark_presidio.py
    uv run python scripts/benchmark_presidio.py --no-spacy-recognizer  # matches plan.md
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poc_classifier import Privacy as POCPrivacy  # noqa: E402
from poc_classifier import classify_t1
from train_encoder import PRIVACIES, PRV2IDX, SEED, load_joined_rows  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONF_IDX = PRV2IDX["confidential"]
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 1200
PRESIDIO_MAX_CHARS = 5000  # cap to control latency on long prompts


def build_analyzer(remove_spacy_recognizer: bool) -> AnalyzerEngine:
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    if remove_spacy_recognizer:
        # Per plan.md: Presidio's NER-driven recognizer is noisy + expensive; disable
        analyzer.registry.remove_recognizer("SpacyRecognizer")
    return analyzer


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
    parser.add_argument("--no-spacy-recognizer", action="store_true",
                        help="Disable SpacyRecognizer per plan.md (pattern-only + non-NER recognizers)")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    print(f"Building Presidio AnalyzerEngine (spacy_recognizer={'OFF' if args.no_spacy_recognizer else 'ON'})...")
    analyzer = build_analyzer(remove_spacy_recognizer=args.no_spacy_recognizer)
    print(f"  Registered recognizers: {len(analyzer.registry.recognizers)}")

    rows = load_joined_rows()
    y_p = np.array([r.privacy for r in rows])
    texts = [r.text for r in rows]
    n = len(rows)
    print(f"\nLoaded {n} rows  (privacy: {dict(Counter(PRIVACIES[p] for p in y_p))})")

    # --- Presidio ---
    print("\nRunning Presidio on all rows (collecting per-request latency)...")
    presidio_conf = np.zeros(n, dtype=bool)
    latencies_ms: list[float] = []
    entity_hits: Counter = Counter()
    for i, t in enumerate(texts):
        t_trunc = t[:PRESIDIO_MAX_CHARS]
        t0 = time.perf_counter()
        results = analyzer.analyze(text=t_trunc, language=args.language)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if results:
            presidio_conf[i] = True
            for r in results:
                entity_hits[r.entity_type] += 1
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{n}  (p50 so far {np.median(latencies_ms):.1f}ms)")

    # --- POC ---
    print("\nRunning POC Tier 1 classifier...")
    poc_conf = np.zeros(n, dtype=bool)
    for i, t in enumerate(texts):
        poc_conf[i] = (classify_t1(t).privacy == POCPrivacy.confidential)

    # --- Recipe B k-fold ---
    print(f"\nEncoding with {EMBED_MODEL} and running Recipe B 5-fold OOF...")
    embed = SentenceTransformer(EMBED_MODEL)
    X = embed.encode([t[:MAX_CHARS] for t in texts],
                     batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    enc_pred = np.full(n, -1, dtype=int)
    for fold, (tr, te) in enumerate(skf.split(X, y_p), 1):
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            random_state=SEED, solver="lbfgs",
        ).fit(X[tr], y_p[tr])
        enc_pred[te] = clf.predict(X[te])
    enc_conf = enc_pred == CONF_IDX

    # --- Union + diagnostics ---
    conf_mask = y_p == CONF_IDX
    gt_conf = int(conf_mask.sum())

    sep = "=" * 74
    print(f"\n{sep}\n  Presidio benchmark + 3-way union gate check  (n={n}, gt_conf={gt_conf})\n{sep}")

    # Latency section
    lat = np.array(latencies_ms)
    print("\n  Presidio per-request latency")
    print(f"    mean: {lat.mean():.1f}ms   p50: {np.percentile(lat, 50):.1f}ms   "
          f"p95: {np.percentile(lat, 95):.1f}ms   p99: {np.percentile(lat, 99):.1f}ms   "
          f"max: {lat.max():.1f}ms")
    p95 = np.percentile(lat, 95)
    if p95 <= 30:
        print("    p95 <= 30ms -> KEEP as parallel Tier 2b per plan.md")
    else:
        print("    p95  > 30ms -> DEMOTE to conditional Tier 3 per plan.md")

    # Top Presidio entity types (informative)
    print("\n  Presidio top entity types hit (any confidence):")
    for ent, cnt in entity_hits.most_common(10):
        print(f"    {ent:30s} {cnt}")

    def report(name: str, flag: np.ndarray):
        tp = int((flag & conf_mask).sum())
        fp = int((flag & ~conf_mask).sum())
        predicted = int(flag.sum())
        pp, plo, phi = wilson_ci(tp, gt_conf)
        print(f"\n  {name}")
        print(f"    Confidential recall: {tp}/{gt_conf} = {pp*100:.2f}%   CI [{plo*100:.2f}, {phi*100:.2f}]")
        print(f"    Gate >= 95%:         {gate_verdict(plo, phi, 0.95)}")
        if predicted:
            print(f"    Total flagged: {predicted} (TP={tp}, FP={fp}, precision={tp/predicted*100:.1f}%)")
        else:
            print("    Total flagged: 0")

    report("POC Tier 1 alone", poc_conf)
    report("Recipe B encoder alone (k-fold)", enc_conf)
    report("Presidio alone", presidio_conf)
    report("UNION(POC + Encoder)",                poc_conf | enc_conf)
    report("UNION(POC + Presidio)",               poc_conf | presidio_conf)
    report("UNION(Encoder + Presidio)",           enc_conf | presidio_conf)
    report("UNION(POC + Encoder + Presidio)",     poc_conf | enc_conf | presidio_conf)

    # Overlap analysis on ground-truth confidentials
    only_presidio  = int(((presidio_conf & ~poc_conf & ~enc_conf) & conf_mask).sum())
    caught_full    = int(((poc_conf | enc_conf | presidio_conf) & conf_mask).sum())
    still_missed   = gt_conf - caught_full
    print(f"\n  Breakdown of {gt_conf} ground-truth confidentials:")
    print(f"    caught by POC+Encoder (previous best): {int(((poc_conf | enc_conf) & conf_mask).sum())}")
    print(f"    Presidio-only additions:                {only_presidio}")
    print(f"    caught by full UNION:                   {caught_full}")
    print(f"    still missed by everything:             {still_missed}  (Tier 3 LLM territory)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
