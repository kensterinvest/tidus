#!/usr/bin/env python3
"""Phase 0 Step 1 backtest — run the POC classifier against WildChat hand-labels.

Gate metrics (from plan.md / boot):
- Privacy: confidential RECALL >= 95%
- Domain:  overall accuracy     >= 85%

Overall privacy accuracy is NOT the gate. The POC's detect_privacy only emits
{internal, confidential} (never `public`), so on a label set that is 85% public,
overall privacy accuracy is ~hard-capped at 15% by construction. The compliance
SLO cares about the asymmetric direction: does the POC correctly flag content
that is confidential.

Pre-committed decision rule (per advisor):
    Wilson 95% CI lower bound >= gate  -> PASS  (skip encoder training)
    Wilson 95% CI upper bound <  gate  -> FAIL  (Phase 1 encoder training)
    CI straddles gate                  -> INCONCLUSIVE (keep labeling)

Reuses (does not modify) scripts/poc_classifier.py.

Usage:
    uv run python scripts/backtest_poc_wildchat.py
    uv run python scripts/backtest_poc_wildchat.py --no-embedding
    uv run python scripts/backtest_poc_wildchat.py --eyeball 10
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Windows default cp950/cp1252 consoles choke on CJK in --eyeball output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poc_classifier import (
    Complexity,
    Domain,
    Privacy,
    classify_t1,
    classify_t2,
    load_embeddings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = REPO_ROOT / "tests" / "classification" / "chunks"
POOL_DIR = REPO_ROOT / "tests" / "classification" / "pool_chunks"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Returns (p_hat, lower_bound, upper_bound) at confidence z (default 95%)."""
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
    return f"INCONCLUSIVE (CI [{lo*100:.1f}, {hi*100:.1f}] straddles {threshold*100:.0f}% — keep labeling)"


def load_labeled_pool() -> tuple[list[tuple[str, str, Domain, Complexity, Privacy]], int, int]:
    """Returns (rows, total_label_rows, unmatched_count).

    Joins every labels_*.jsonl row against pool_*.jsonl by id.
    """
    pool_text: dict[str, str] = {}
    for pf in sorted(POOL_DIR.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                pool_text[r["id"]] = r["text"]

    rows = []
    total = 0
    unmatched = 0
    for lf in sorted(CHUNKS_DIR.glob("labels_*.jsonl")):
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                total += 1
                if r["id"] not in pool_text:
                    unmatched += 1
                    continue
                rows.append((
                    r["id"],
                    pool_text[r["id"]],
                    Domain(r["domain"]),
                    Complexity(r["complexity"]),
                    Privacy(r["privacy"]),
                ))
    return rows, total, unmatched


def print_confusion(title: str, enum_cls, cm: Counter) -> None:
    print(f"\n  {title} confusion (rows=actual, cols=predicted)")
    vals = [v.value for v in enum_cls]
    print("    " + " ".join(f"{v[:7]:>7}" for v in vals) + "   | total")
    for actual in vals:
        row_total = sum(cm.get((actual, p), 0) for p in vals)
        row = f"    {actual[:13]:>13}"
        for pred in vals:
            c = cm.get((actual, pred), 0)
            row += f" {c:>6d} " if c else "      . "
        print(row + f"  | {row_total}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embedding", action="store_true", help="Tier 1 only")
    parser.add_argument("--eyeball", type=int, default=0,
                        help="Print first N confidential-recall misses for manual check")
    args = parser.parse_args()

    use_t2 = not args.no_embedding
    if use_t2:
        print("Loading all-MiniLM-L6-v2 ...")
        if not load_embeddings():
            print("  sentence-transformers not installed -- running Tier 1 only.")
            use_t2 = False
        else:
            print("  Model ready.")

    rows, total_labels, unmatched = load_labeled_pool()
    n = len(rows)
    print(f"\nJoined {n}/{total_labels} label rows to pool text  ({unmatched} unmatched — expected for labels_001-009 from prior dataset).")
    print(f"Tier 2: {'enabled' if use_t2 else 'disabled'}\n")

    dom_ok = 0
    cmp_ok = 0
    priv_overall_ok = 0
    conf_ground = 0
    conf_flagged = 0
    conf_predicted = 0
    internal_ground = 0
    internal_to_conf = 0

    dom_cm: Counter = Counter()
    cmp_cm: Counter = Counter()
    priv_cm: Counter = Counter()

    eyeball_misses: list[tuple[str, str, str]] = []

    for cid, text, exp_dom, exp_cmp, exp_priv in rows:
        t1 = classify_t1(text)
        res = classify_t2(text, t1) if use_t2 else t1

        dom_ok += int(res.domain == exp_dom)
        cmp_ok += int(res.complexity == exp_cmp)
        priv_overall_ok += int(res.privacy == exp_priv)

        dom_cm[(exp_dom.value, res.domain.value)] += 1
        cmp_cm[(exp_cmp.value, res.complexity.value)] += 1
        priv_cm[(exp_priv.value, res.privacy.value)] += 1

        if exp_priv == Privacy.confidential:
            conf_ground += 1
            if res.privacy == Privacy.confidential:
                conf_flagged += 1
            elif args.eyeball and len(eyeball_misses) < args.eyeball:
                eyeball_misses.append((cid, text[:200], res.privacy.value))
        if exp_priv == Privacy.internal:
            internal_ground += 1
            if res.privacy == Privacy.confidential:
                internal_to_conf += 1
        if res.privacy == Privacy.confidential:
            conf_predicted += 1

    sep = "=" * 72
    print(sep)
    print(f"  Tidus POC Backtest vs WildChat hand-labels  (n={n})")
    print(f"  Tier 2: {'enabled' if use_t2 else 'disabled'}")
    print(sep)

    dom_p, dom_lo, dom_hi = wilson_ci(dom_ok, n)
    print("\n  Domain accuracy")
    print(f"    {dom_ok}/{n} = {dom_p*100:.2f}%   95% CI [{dom_lo*100:.2f}, {dom_hi*100:.2f}]")
    print(f"    Gate >= 85%  -> {gate_verdict(dom_lo, dom_hi, 0.85)}")

    cmp_p, cmp_lo, cmp_hi = wilson_ci(cmp_ok, n)
    print("\n  Complexity accuracy  (no gate — reported only)")
    print(f"    {cmp_ok}/{n} = {cmp_p*100:.2f}%   95% CI [{cmp_lo*100:.2f}, {cmp_hi*100:.2f}]")

    # Privacy gate: CONFIDENTIAL RECALL
    conf_p, conf_lo, conf_hi = wilson_ci(conf_flagged, conf_ground)
    print("\n  Privacy — CONFIDENTIAL RECALL  (the SLO-gate metric)")
    print(f"    {conf_flagged}/{conf_ground} = {conf_p*100:.2f}%   95% CI [{conf_lo*100:.2f}, {conf_hi*100:.2f}]")
    print(f"    Gate >= 95%  -> {gate_verdict(conf_lo, conf_hi, 0.95)}")

    # Secondary: confidential precision + overall privacy
    if conf_predicted:
        prec = conf_flagged / conf_predicted
        print(f"\n  Privacy — confidential precision (secondary)")
        print(f"    {conf_flagged}/{conf_predicted} = {prec*100:.2f}%   ({conf_predicted - conf_flagged} false-confidentials)")
    if internal_ground:
        esc = internal_to_conf / internal_ground
        print(f"\n  Privacy — internal->confidential escalation rate (context)")
        print(f"    {internal_to_conf}/{internal_ground} = {esc*100:.2f}%")

    po_p, po_lo, po_hi = wilson_ci(priv_overall_ok, n)
    print("\n  Privacy overall accuracy (context — POC never emits 'public', so capped ~15%)")
    print(f"    {priv_overall_ok}/{n} = {po_p*100:.2f}%")

    print_confusion("Domain", Domain, dom_cm)
    print_confusion("Complexity", Complexity, cmp_cm)
    print_confusion("Privacy", Privacy, priv_cm)

    dom_pass = dom_lo >= 0.85
    dom_fail = dom_hi < 0.85
    conf_pass = conf_lo >= 0.95
    conf_fail = conf_hi < 0.95

    print("\n" + sep)
    print("  Phase 0 gate decision (pre-committed rule)")
    print(f"    Domain  >= 85% : {'PASS' if dom_pass else ('FAIL' if dom_fail else 'INCONCLUSIVE')}")
    print(f"    Privacy >= 95% : {'PASS' if conf_pass else ('FAIL' if conf_fail else 'INCONCLUSIVE')}")
    if dom_pass and conf_pass:
        print("  -> OVERALL PASS: encoder training may be skipped.")
        ret = 0
    elif dom_fail or conf_fail:
        print("  -> OVERALL FAIL: proceed to Phase 1 encoder training.")
        ret = 1
    else:
        print("  -> OVERALL INCONCLUSIVE: keep labeling until CI tightens.")
        ret = 2
    print(sep)

    if args.eyeball and eyeball_misses:
        print(f"\n  First {len(eyeball_misses)} confidential-recall misses (eyeball check):")
        for cid, txt, got in eyeball_misses:
            preview = " ".join(txt.split())[:160]
            print(f"\n    [{cid}]  POC: {got}")
            print(f"      {preview}")

    return ret


if __name__ == "__main__":
    raise SystemExit(main())
