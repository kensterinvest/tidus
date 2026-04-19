#!/usr/bin/env python3
"""Inspect ground-truth confidentials that EVERY signal misses (E1 ceiling).

The ensemble sweep found a plateau at 54/57. Rules E1, E3, E4, E7 all hit the
same ceiling — meaning 3 confidentials escape POC AND Encoder AND high-trust
Presidio AND PERSON NER. Before accepting "these need Tier 3 LLM", re-audit:
the prior audit round found 28% overcall rate, and these 3 have only been
reviewed at population level, not individually.

Writes a text report with full preview so the human reviewer can flip/keep/ambig.
"""
from __future__ import annotations

import json
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

REPO = Path(__file__).resolve().parent.parent
CHUNKS = REPO / "tests" / "classification" / "chunks"
POOL = REPO / "tests" / "classification" / "pool_chunks"
CONF_IDX = PRV2IDX["confidential"]
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


def main() -> int:
    rows = load_joined_rows()
    texts = [r.text for r in rows]
    y_p = np.array([r.privacy for r in rows])
    n = len(rows)

    # Map row index -> (id, chunk, rationale)
    pool_text = {}
    for pf in sorted(POOL.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                pool_text[r["id"]] = r["text"]
    id_list: list[tuple[str, str, str]] = []
    for lf in sorted(CHUNKS.glob("labels_*.jsonl")):
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["id"] in pool_text:
                    id_list.append((r["id"], lf.stem, r.get("rationale", "")))
    assert len(id_list) == n

    print(f"Loaded {n} rows")
    print("POC Tier 1...")
    poc_conf = np.zeros(n, dtype=bool)
    for i, t in enumerate(texts):
        poc_conf[i] = (classify_t1(t).privacy == POCPrivacy.confidential)

    print("Encoder 5-fold OOF...")
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

    print("Presidio NER-on scan...")
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    per_types: list[set[str]] = []
    for i, t in enumerate(texts):
        results = analyzer.analyze(text=t[:PRESIDIO_MAX_CHARS], language="en")
        per_types.append({r.entity_type for r in results})
        if (i + 1) % 400 == 0:
            print(f"  ...{i+1}/{n}")

    has_high_trust = np.array([bool(t & HIGH_TRUST) for t in per_types], dtype=bool)
    has_person = np.array(["PERSON" in t for t in per_types], dtype=bool)

    # E1 union: POC | Encoder | HighTrust | PERSON
    e1_caught = poc_conf | enc_conf | has_high_trust | has_person
    conf_mask = y_p == CONF_IDX
    all_missed = conf_mask & ~e1_caught
    idxs = np.where(all_missed)[0]

    print(f"\nAll-signal-missed confidentials: {len(idxs)}")

    report = []
    report.append("# Confidentials missed by EVERY signal (E1 ceiling)")
    report.append(f"# n_total={n}, gt_conf={int(conf_mask.sum())}, all_missed={len(idxs)}")
    report.append("")
    report.append("For each, decide BEFORE checking impact on metrics:")
    report.append("  KEEP   = genuinely confidential, no detector on earth would have")
    report.append("           caught it without semantic reasoning -> Tier 3 LLM territory")
    report.append("  FLIP   = labeler overcall -> add to label_overrides.jsonl")
    report.append("  AMBIG  = taxonomy edge case, reasonable people disagree")
    report.append("")
    report.append("=" * 80)
    for idx in idxs:
        cid, chunk, rationale = id_list[idx]
        text = texts[idx]
        preview_800 = " ".join(text.split())[:800]
        entities_seen = sorted(per_types[idx])
        report.append("")
        report.append(f"[{chunk}] {cid}")
        report.append(f"  rationale: {rationale}")
        report.append(f"  text_len:  {len(text)}")
        report.append(f"  presidio_entities_seen: {entities_seen if entities_seen else '(none)'}")
        report.append(f"  poc_caught:  {bool(poc_conf[idx])}")
        report.append(f"  enc_caught:  {bool(enc_conf[idx])}")
        report.append(f"  text[:800]:  {preview_800}")

    out_path = REPO / "audit_all_missed.txt"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
