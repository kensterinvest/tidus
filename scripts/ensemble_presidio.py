#!/usr/bin/env python3
"""Ensemble rule sweep — use Presidio PERSON as a CORROBORATING signal only.

tune_presidio.py findings: `high-trust + PERSON` gives 94.7% union recall but
50% traffic flagged (precision 7%). Each 1pp recall gain from PERSON costs
3pp of flagged traffic. PERSON is noisy — most hits are benign capitalized
words, not real PII.

This script tests ENSEMBLE rules where PERSON only triggers `confidential`
when CORROBORATED by another signal (encoder's "not public" prediction,
a second Presidio entity, etc.). High-trust Presidio recognizers (PHONE,
EMAIL, SSN, etc.) still trigger standalone. POC + Encoder already form
the union baseline.

Rules tested:
    E1  PERSON alone trigger (baseline = high-trust + PERSON from tune)
    E2  PERSON triggers ONLY if Encoder says "internal" or "confidential"
    E3  PERSON triggers ONLY if >= 1 other Presidio entity also hits (any)
    E4  PERSON triggers ONLY if its detection score >= threshold (0.85 default)
    E5  PERSON triggers ONLY if >= 2 separate PERSON spans in same text
    E6  PERSON triggers ONLY if (Encoder non-public AND other Presidio entity)
    E7  NO PERSON (baseline = high-trust only)

All rules sit ON TOP of the high-trust union = POC ∪ Encoder(conf) ∪ HighTrustPresidio.

Usage:
    uv run python scripts/ensemble_presidio.py
"""
from __future__ import annotations

import math
import sys
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
from train_encoder import PRV2IDX, SEED, load_joined_rows  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONF_IDX = PRV2IDX["confidential"]
PUB_IDX = PRV2IDX["public"]
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 1200
PRESIDIO_MAX_CHARS = 5000

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


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - halfw), min(1.0, center + halfw)


def build_analyzer_ner_on() -> AnalyzerEngine:
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine)


def scan_presidio(analyzer, texts: list[str]) -> tuple[list[set[str]], list[list[dict]]]:
    """Returns (per_row_entity_types, per_row_hit_details).

    hit_details = [{entity_type, score, start, end}, ...] for each row — so we
    can threshold by score, count duplicate entity types, etc.
    """
    per_row_types: list[set[str]] = []
    per_row_details: list[list[dict]] = []
    for i, t in enumerate(texts):
        results = analyzer.analyze(text=t[:PRESIDIO_MAX_CHARS], language="en")
        per_row_types.append({r.entity_type for r in results})
        per_row_details.append([
            {"entity_type": r.entity_type, "score": float(r.score),
             "start": r.start, "end": r.end} for r in results
        ])
        if (i + 1) % 200 == 0:
            print(f"    ...{i+1}/{len(texts)}")
    return per_row_types, per_row_details


def evaluate(
    name: str,
    presidio_conf: np.ndarray,
    poc_conf: np.ndarray,
    enc_conf: np.ndarray,
    conf_mask: np.ndarray,
    n: int,
) -> dict:
    full = poc_conf | enc_conf | presidio_conf
    gt_conf = int(conf_mask.sum())
    tp = int((full & conf_mask).sum())
    flagged = int(full.sum())
    p, lo, hi = wilson_ci(tp, gt_conf)
    return {
        "name": name,
        "recall": p,
        "ci": (lo, hi),
        "tp": tp,
        "flagged_pct": flagged / n * 100,
        "precision": tp / flagged * 100 if flagged else 0.0,
    }


def main() -> int:
    rows = load_joined_rows()
    y_p = np.array([r.privacy for r in rows])
    texts = [r.text for r in rows]
    n = len(rows)
    conf_mask = y_p == CONF_IDX
    gt_conf = int(conf_mask.sum())
    print(f"Loaded {n} rows, gt_conf={gt_conf}")

    # POC
    print("\nPOC Tier 1...")
    poc_conf = np.zeros(n, dtype=bool)
    for i, t in enumerate(texts):
        poc_conf[i] = (classify_t1(t).privacy == POCPrivacy.confidential)

    # Encoder — we need FULL 3-class predictions (not just binary conf) for rule E2
    print(f"\nEncoder 5-fold OOF with {EMBED_MODEL}...")
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
    enc_nonpublic = enc_pred != PUB_IDX  # True if encoder says internal or confidential

    # Presidio (NER on — we need PERSON details)
    print("\nPresidio NER-on scan...")
    analyzer = build_analyzer_ner_on()
    per_types, per_details = scan_presidio(analyzer, texts)

    # --- Per-row flags ---
    has_high_trust = np.array([bool(t & HIGH_TRUST) for t in per_types], dtype=bool)
    has_person = np.array(["PERSON" in t for t in per_types], dtype=bool)
    person_count = np.array([sum(1 for h in d if h["entity_type"] == "PERSON") for d in per_details])
    max_person_score = np.array([
        max((h["score"] for h in d if h["entity_type"] == "PERSON"), default=0.0)
        for d in per_details
    ])
    # count non-PERSON entities in each row
    other_entity_count = np.array([
        len([h for h in d if h["entity_type"] not in {"PERSON"}])
        for d in per_details
    ])

    # --- Rule evaluation ---
    # Each rule produces a presidio_conf mask; union with POC + encoder is done in evaluate().
    rules = []

    # E0 baseline: NO Presidio (just POC + Encoder) — reference
    rules.append(("E0 POC+Encoder only (no Presidio)",
                  np.zeros(n, dtype=bool)))

    # E-baseline: high-trust only (no PERSON at all)
    rules.append(("high-trust Presidio only (no PERSON)",
                  has_high_trust.copy()))

    # E1: PERSON alone triggers (matches tune_presidio.py high-trust+PERSON)
    rules.append(("E1 PERSON alone triggers",
                  has_high_trust | has_person))

    # E2: PERSON only when Encoder says non-public (corroboration)
    e2_flag = has_high_trust | (has_person & enc_nonpublic)
    rules.append(("E2 PERSON + Encoder-non-public", e2_flag))

    # E3: PERSON only when >= 1 other Presidio entity
    e3_flag = has_high_trust | (has_person & (other_entity_count >= 1))
    rules.append(("E3 PERSON + >= 1 other Presidio entity", e3_flag))

    # E4 variants: PERSON score thresholds
    for thresh in (0.60, 0.75, 0.85, 0.95):
        e4_flag = has_high_trust | (has_person & (max_person_score >= thresh))
        rules.append((f"E4 PERSON score >= {thresh:.2f}", e4_flag))

    # E5: PERSON only when >= 2 separate PERSON spans in same text
    e5_flag = has_high_trust | (person_count >= 2)
    rules.append(("E5 PERSON count >= 2", e5_flag))

    # E6: PERSON + (Encoder-non-public AND another Presidio entity)
    e6_flag = has_high_trust | (has_person & enc_nonpublic & (other_entity_count >= 1))
    rules.append(("E6 PERSON + Enc-non-pub + other entity", e6_flag))

    # E7: PERSON with high score OR corroborated by Encoder-non-public
    e7_flag = has_high_trust | (has_person & ((max_person_score >= 0.85) | enc_nonpublic))
    rules.append(("E7 PERSON (score>=0.85 OR Enc-non-pub)", e7_flag))

    # Evaluate all
    results = [evaluate(nm, p_conf, poc_conf, enc_conf, conf_mask, n) for nm, p_conf in rules]

    sep = "=" * 110
    print(f"\n{sep}")
    print(f"  Ensemble rule sweep  (n={n}, gt_conf={gt_conf}, POC+Encoder baseline = 79.0% / 10% flagged)")
    print(sep)
    print(f"\n  {'Rule':<44} {'UNION recall':<26} {'Flagged%':<10} {'Precision':<10} {'TP'}")
    print(f"  {'-'*44} {'-'*26} {'-'*10} {'-'*10} {'-'*4}")
    for r in results:
        ci_str = f"{r['recall']*100:5.1f}% [{r['ci'][0]*100:4.1f},{r['ci'][1]*100:5.1f}]"
        print(f"  {r['name']:<44} {ci_str:<26} {r['flagged_pct']:5.1f}%    {r['precision']:5.1f}%    {r['tp']}/{gt_conf}")

    print("\n  Gate check (union recall ≥ 95% — CI lower bound):")
    for r in results:
        lo = r['ci'][0]
        hi = r['ci'][1]
        if lo >= 0.95:
            print(f"    PASS      {r['name']:<44} CI lower {lo*100:.1f}%")
        elif hi >= 0.95:
            print(f"    INCONCL   {r['name']:<44} CI [{lo*100:.1f}, {hi*100:.1f}], flagged {r['flagged_pct']:.1f}%")

    # Pareto: at each flagged% threshold, best recall
    print("\n  Pareto frontier (best-recall config at each traffic-flagged budget):")
    for budget in (10, 15, 20, 25, 30, 40, 50, 75):
        best = max((r for r in results if r['flagged_pct'] <= budget), key=lambda x: x['recall'], default=None)
        if best:
            print(f"    ≤{budget}%% traffic: {best['name']:<44} {best['recall']*100:5.1f}% [{best['ci'][0]*100:.1f}, {best['ci'][1]*100:.1f}]")
        else:
            print(f"    <={budget}%% traffic: no config found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
