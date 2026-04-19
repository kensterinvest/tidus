#!/usr/bin/env python3
"""Presidio recognizer tuning — find Pareto-optimal entity allowlist.

Benchmark (benchmark_presidio.py) showed Presidio's full recognizer set is
high-recall (98%) but has catastrophic precision (77% of ALL traffic flagged).
Dominant false-positive sources: URL (3949 hits), US_DRIVER_LICENSE (1612),
US_BANK_NUMBER (792), plus NER entities (ORG, PERSON, DATE_TIME, LOCATION).

This script runs Presidio once per NER config, collects per-row per-entity
hits, then evaluates multiple allowlist configurations against the same data
(near-instant after the initial scan). Reports recall/precision/traffic-flagged
plus union recall with POC+Encoder for each config.

Usage:
    uv run python scripts/tune_presidio.py
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poc_classifier import Privacy as POCPrivacy, classify_t1  # noqa: E402
from train_encoder import PRIVACIES, PRV2IDX, SEED, load_joined_rows  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONF_IDX = PRV2IDX["confidential"]
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 1200
PRESIDIO_MAX_CHARS = 5000


def build_analyzer(remove_spacy_recognizer: bool) -> AnalyzerEngine:
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    if remove_spacy_recognizer:
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


def scan_presidio(analyzer, texts: list[str]) -> list[set[str]]:
    """Run Presidio on all rows, return per-row set of entity types seen."""
    per_row: list[set[str]] = []
    for i, t in enumerate(texts):
        results = analyzer.analyze(text=t[:PRESIDIO_MAX_CHARS], language="en")
        per_row.append({r.entity_type for r in results})
        if (i + 1) % 200 == 0:
            print(f"    ...{i+1}/{len(texts)}")
    return per_row


def eval_config(
    name: str,
    allow: set[str],
    per_row: list[set[str]],
    poc_conf: np.ndarray,
    enc_conf: np.ndarray,
    conf_mask: np.ndarray,
) -> dict:
    """For each row: flagged if any entity in row is in allowlist."""
    n = len(per_row)
    presidio_conf = np.array([bool(s & allow) for s in per_row], dtype=bool)
    gt_conf = int(conf_mask.sum())

    full = poc_conf | enc_conf | presidio_conf
    tp_full = int((full & conf_mask).sum())
    flagged_full = int(full.sum())
    p_full, lo_full, hi_full = wilson_ci(tp_full, gt_conf)

    tp_p = int((presidio_conf & conf_mask).sum())
    p_alone, lo_alone, hi_alone = wilson_ci(tp_p, gt_conf)
    flagged_p = int(presidio_conf.sum())

    return {
        "name": name,
        "allow_count": len(allow),
        # Presidio alone
        "p_recall": p_alone,
        "p_ci": (lo_alone, hi_alone),
        "p_flagged_pct": flagged_p / n * 100,
        "p_precision": tp_p / flagged_p * 100 if flagged_p else 0.0,
        # Full union (POC + Encoder + Presidio)
        "union_recall": p_full,
        "union_ci": (lo_full, hi_full),
        "union_flagged_pct": flagged_full / n * 100,
        "union_precision": tp_full / flagged_full * 100 if flagged_full else 0.0,
        "union_tp": tp_full,
    }


def main() -> int:
    rows = load_joined_rows()
    y_p = np.array([r.privacy for r in rows])
    texts = [r.text for r in rows]
    n = len(rows)
    conf_mask = y_p == CONF_IDX
    gt_conf = int(conf_mask.sum())
    print(f"Loaded {n} rows, gt_conf={gt_conf}")

    # --- POC + Encoder signals (same as backtest_union.py) ---
    print("\nPOC Tier 1...")
    poc_conf = np.zeros(n, dtype=bool)
    for i, t in enumerate(texts):
        poc_conf[i] = (classify_t1(t).privacy == POCPrivacy.confidential)

    print(f"\nRecipe B 5-fold OOF with {EMBED_MODEL}...")
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

    # --- Presidio: two scans (NER on, NER off) ---
    print("\nPresidio scan 1/2: NER ON (spaCy recognizer enabled)...")
    analyzer_on = build_analyzer(remove_spacy_recognizer=False)
    per_row_on = scan_presidio(analyzer_on, texts)

    print("\nPresidio scan 2/2: NER OFF (pattern + non-NER recognizers only)...")
    analyzer_off = build_analyzer(remove_spacy_recognizer=True)
    per_row_off = scan_presidio(analyzer_off, texts)

    # --- Configurations to evaluate ---
    # Pattern-only genuine PII (high-trust)
    HIGH_TRUST = {
        "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "IP_ADDRESS",
        "UK_NHS", "CREDIT_CARD", "US_PASSPORT", "MEDICAL_LICENSE",
        "IBAN_CODE", "CRYPTO", "AU_ABN", "AU_ACN", "AU_TFN",
        "AU_MEDICARE", "ES_NIF", "IT_DRIVER_LICENSE", "IT_FISCAL_CODE",
        "IT_VAT_CODE", "IT_PASSPORT", "IT_IDENTITY_CARD",
        "SG_NRIC_FIN", "SG_UEN", "PL_PESEL", "KR_RRN", "IN_AADHAAR",
        "IN_VEHICLE_REGISTRATION", "IN_VOTER", "IN_PASSPORT", "IN_PAN",
        "FI_PERSONAL_IDENTITY_CODE", "NG_NIN",
    }
    NOISY_PATTERNS = {"URL", "US_DRIVER_LICENSE", "US_BANK_NUMBER", "DATE_TIME", "US_ITIN"}
    NER_ENTITIES = {"PERSON", "ORGANIZATION", "LOCATION", "NRP"}
    # Union of everything Presidio can emit (for "baseline = all" configs)
    all_entities_on = set().union(*per_row_on) if per_row_on else set()
    all_entities_off = set().union(*per_row_off) if per_row_off else set()

    configs = [
        # Baselines (match benchmark_presidio.py output)
        ("NER-on ALL (baseline)",       all_entities_on,          per_row_on),
        ("NER-off ALL (plan.md)",       all_entities_off,         per_row_off),
        # Pattern-only tightening (off the NER-off scan)
        ("NER-off, drop URL",           all_entities_off - {"URL"}, per_row_off),
        ("NER-off, drop URL+DL",        all_entities_off - {"URL", "US_DRIVER_LICENSE"}, per_row_off),
        ("NER-off, drop URL+DL+BANK",   all_entities_off - {"URL", "US_DRIVER_LICENSE", "US_BANK_NUMBER"}, per_row_off),
        ("NER-off, drop all noisy",     all_entities_off - NOISY_PATTERNS, per_row_off),
        ("high-trust only",             HIGH_TRUST,               per_row_off),
        # With PERSON added back (off NER-on scan)
        ("high-trust + PERSON",         HIGH_TRUST | {"PERSON"},  per_row_on),
        ("NER-off noisy-dropped + PERSON", (all_entities_off - NOISY_PATTERNS) | {"PERSON"}, per_row_on),
    ]

    results = []
    for name, allow, per_row in configs:
        r = eval_config(name, allow, per_row, poc_conf, enc_conf, conf_mask)
        results.append(r)

    # --- Report ---
    sep = "=" * 110
    print(f"\n{sep}")
    print(f"  Presidio tuning — Pareto sweep  (n={n}, gt_conf={gt_conf}, POC+Encoder baseline = 45/57 = 79.0%)")
    print(sep)
    print(f"\n  {'Config':<38} {'Presidio alone':<28} {'UNION(POC+Enc+Presidio)':<34} {'Traffic%'}")
    print(f"  {'-'*38} {'-'*28} {'-'*34} {'-'*8}")
    for r in results:
        p_str = f"{r['p_recall']*100:5.1f}% [{r['p_ci'][0]*100:4.1f},{r['p_ci'][1]*100:5.1f}] p{r['p_precision']:.0f}%"
        u_str = f"{r['union_recall']*100:5.1f}% [{r['union_ci'][0]*100:4.1f},{r['union_ci'][1]*100:5.1f}] p{r['union_precision']:.0f}% n={r['union_tp']}"
        print(f"  {r['name']:<38} {p_str:<28} {u_str:<34} {r['union_flagged_pct']:5.1f}%")

    print()
    # Gate summary — only the ones that clear OR are close
    print("  Gate check (union recall ≥ 95% at CI lower bound):")
    for r in results:
        lo = r['union_ci'][0]
        if lo >= 0.95:
            print(f"    PASS  {r['name']:<38} CI lower {lo*100:.1f}%")
        elif r['union_ci'][1] >= 0.95:
            print(f"    INCONCL  {r['name']:<38} CI [{lo*100:.1f}, {r['union_ci'][1]*100:.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
