#!/usr/bin/env python3
"""Score cross-family inter-rater reliability for privacy classification.

Reads:
  - tests/classification/irr/truth_keys.jsonl        (our = "claude" labels)
  - tests/classification/irr/responses/pack_NN_gpt.jsonl
  - tests/classification/irr/responses/pack_NN_gemini.jsonl

Computes:
  - Overall agreement (observed agreement fraction)
  - Pairwise Cohen's kappa   (Claude-GPT, Claude-Gemini, GPT-Gemini)
  - Fleiss' kappa            (3-rater agreement across the whole sample)
  - Per-class recall of non-Claude labelers against Claude "confidential"
  - Confusion matrices per pair
  - Disagreement list (rows where any pair disagrees on privacy — for adjudication)
  - Structural-miss verdict (did GPT/Gemini catch the 3 audit cases?)

Writes markdown report to tests/classification/irr/irr_report.md.

Run:  uv run python scripts/irr_score.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IRR_DIR = ROOT / "tests" / "classification" / "irr"
RESP_DIR = IRR_DIR / "responses"

PRIVACY_CLASSES = ["public", "internal", "confidential"]

AUDIT_CASES = {
    "wildchat-3aba9fbe9541a6262ea606a7af9fd328": "Case 1 (Vue/SCSS Chinese placeholder — flipped to public)",
    "wildchat-d5a328a8e0a5c8e7215ea2bd01ad8eff": "Case 2 (Canadian work permit — kept confidential)",
    "wildchat-1b26ab6746ab5e4178abe77c22858085": "Case 3 (Russian mental health — kept confidential)",
}


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", line)
            if json_match:
                try:
                    out.append(json.loads(json_match.group(0)))
                except json.JSONDecodeError:
                    continue
    return out


def load_labeler(patterns: list[str]) -> dict[str, str]:
    """Merge all pack_*_{labeler}.* files into {id: privacy}.
    Accepts multiple glob patterns to tolerate filename variants
    (.jsonl vs .jsonl.md; gemini vs genmini spelling)."""
    merged: dict[str, str] = {}
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(RESP_DIR.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            for row in load_jsonl(path):
                rid = row.get("id")
                privacy = row.get("privacy")
                if rid and privacy in PRIVACY_CLASSES:
                    merged[rid] = privacy
    return merged


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa (unweighted) between two equal-length label lists."""
    assert len(labels_a) == len(labels_b) and labels_a
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n

    counts_a: dict[str, int] = defaultdict(int)
    counts_b: dict[str, int] = defaultdict(int)
    for a, b in zip(labels_a, labels_b):
        counts_a[a] += 1
        counts_b[b] += 1
    p_e = sum((counts_a[c] / n) * (counts_b[c] / n) for c in set(counts_a) | set(counts_b))

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def weighted_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Quadratic-weighted Cohen's kappa, appropriate for ordinal classes.

    For privacy: public < internal < confidential (distance 2 between public
    and confidential, distance 1 for adjacent classes). Penalizes distant
    disagreements more than adjacent ones — the right metric when labels have
    a natural ordering.
    """
    order = {c: i for i, c in enumerate(PRIVACY_CLASSES)}
    assert len(labels_a) == len(labels_b) and labels_a
    n = len(labels_a)
    k = len(PRIVACY_CLASSES)

    weights = [[((order[a] - order[b]) / (k - 1)) ** 2 for b in PRIVACY_CLASSES] for a in PRIVACY_CLASSES]

    obs = [[0.0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        obs[order[a]][order[b]] += 1
    for i in range(k):
        for j in range(k):
            obs[i][j] /= n

    row_totals = [sum(row) for row in obs]
    col_totals = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    exp = [[row_totals[i] * col_totals[j] for j in range(k)] for i in range(k)]

    num = sum(weights[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(weights[i][j] * exp[i][j] for i in range(k) for j in range(k))
    if den == 0:
        return 1.0
    return 1 - (num / den)


def fleiss_kappa(rater_labels: list[list[str]]) -> float:
    """Fleiss' kappa for k raters over N items. rater_labels[r][i] = rater r's label for item i."""
    k = len(rater_labels)
    n = len(rater_labels[0])
    assert all(len(r) == n for r in rater_labels), "rater rows must be equal length"

    classes = sorted({c for rl in rater_labels for c in rl})

    # P_i for each item: (sum_j n_ij^2 - k) / (k * (k-1))
    per_item_agreement: list[float] = []
    class_totals: dict[str, int] = defaultdict(int)
    for i in range(n):
        item_counts: dict[str, int] = defaultdict(int)
        for r in range(k):
            item_counts[rater_labels[r][i]] += 1
            class_totals[rater_labels[r][i]] += 1
        p_i = (sum(c * c for c in item_counts.values()) - k) / (k * (k - 1))
        per_item_agreement.append(p_i)

    p_bar = sum(per_item_agreement) / n
    total_ratings = n * k
    p_e_bar = sum((class_totals[c] / total_ratings) ** 2 for c in classes)

    if p_e_bar == 1.0:
        return 1.0
    return (p_bar - p_e_bar) / (1 - p_e_bar)


def kappa_interpretation(k: float) -> str:
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "near-perfect"


def confusion_matrix(a_labels: dict[str, str], b_labels: dict[str, str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        c: {cc: 0 for cc in PRIVACY_CLASSES} for c in PRIVACY_CLASSES
    }
    for rid, a in a_labels.items():
        if rid in b_labels:
            matrix[a][b_labels[rid]] += 1
    return matrix


def format_matrix(m: dict[str, dict[str, int]], row_label: str, col_label: str) -> str:
    lines = [f"|  {row_label} \\\\ {col_label} | " + " | ".join(PRIVACY_CLASSES) + " |"]
    lines.append("|" + "---|" * (len(PRIVACY_CLASSES) + 1))
    for rc in PRIVACY_CLASSES:
        row = [str(m[rc][cc]) for cc in PRIVACY_CLASSES]
        lines.append(f"| **{rc}** | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    if not (IRR_DIR / "truth_keys.jsonl").exists():
        raise SystemExit(
            "truth_keys.jsonl missing. Run scripts/irr_build_external_pack.py first."
        )

    claude_labels: dict[str, str] = {}
    for row in load_jsonl(IRR_DIR / "truth_keys.jsonl"):
        claude_labels[row["id"]] = row["privacy"]

    gpt_labels = load_labeler(["pack_*_gpt.jsonl", "pack_*_gpt.jsonl.md", "pack_*_gpt.md"])
    gemini_labels = load_labeler([
        "pack_*_gemini.jsonl", "pack_*_gemini.jsonl.md", "pack_*_gemini.md",
        "pack_*_genmini.jsonl", "pack_*_genmini.jsonl.md", "pack_*_genmini.md",
    ])

    print(f"claude: {len(claude_labels)} labels")
    print(f"gpt:    {len(gpt_labels)} labels")
    print(f"gemini: {len(gemini_labels)} labels")

    common_ids = sorted(set(claude_labels) & set(gpt_labels) & set(gemini_labels))
    print(f"common ids across all 3 raters: {len(common_ids)}")

    if not common_ids:
        raise SystemExit(
            "No common ids. Check responses/ directory for pack_NN_gpt.jsonl and pack_NN_gemini.jsonl files."
        )

    c = [claude_labels[i] for i in common_ids]
    g = [gpt_labels[i] for i in common_ids]
    m = [gemini_labels[i] for i in common_ids]

    kappa_cg = cohens_kappa(c, g)
    kappa_cm = cohens_kappa(c, m)
    kappa_gm = cohens_kappa(g, m)
    kappa_fleiss = fleiss_kappa([c, g, m])

    wkappa_cg = weighted_kappa(c, g)
    wkappa_cm = weighted_kappa(c, m)
    wkappa_gm = weighted_kappa(g, m)

    observed_c_g = sum(1 for a, b in zip(c, g) if a == b) / len(common_ids)
    observed_c_m = sum(1 for a, b in zip(c, m) if a == b) / len(common_ids)
    observed_g_m = sum(1 for a, b in zip(g, m) if a == b) / len(common_ids)
    all_three = sum(1 for a, b, cc in zip(c, g, m) if a == b == cc) / len(common_ids)

    cm_cg = confusion_matrix(
        {i: claude_labels[i] for i in common_ids},
        {i: gpt_labels[i] for i in common_ids},
    )
    cm_cm = confusion_matrix(
        {i: claude_labels[i] for i in common_ids},
        {i: gemini_labels[i] for i in common_ids},
    )
    cm_gm = confusion_matrix(
        {i: gpt_labels[i] for i in common_ids},
        {i: gemini_labels[i] for i in common_ids},
    )

    disagreements: list[dict] = []
    for rid in common_ids:
        labels = (claude_labels[rid], gpt_labels[rid], gemini_labels[rid])
        if len(set(labels)) > 1:
            disagreements.append({
                "id": rid,
                "claude": labels[0],
                "gpt": labels[1],
                "gemini": labels[2],
                "pattern": "3-way" if len(set(labels)) == 3 else "2-1",
                "audit_case": AUDIT_CASES.get(rid),
            })

    majority: dict[str, str] = {}
    for rid in common_ids:
        votes = [claude_labels[rid], gpt_labels[rid], gemini_labels[rid]]
        counts = {c: votes.count(c) for c in set(votes)}
        top = max(counts, key=counts.get)
        if counts[top] >= 2:
            majority[rid] = top
        else:
            majority[rid] = "AMBIG"

    asym_majority: dict[str, str] = {}
    for rid in common_ids:
        votes = {claude_labels[rid], gpt_labels[rid], gemini_labels[rid]}
        if "confidential" in votes:
            asym_majority[rid] = "confidential"
        elif "internal" in votes:
            asym_majority[rid] = "internal"
        else:
            asym_majority[rid] = "public"

    claude_vs_majority = sum(
        1 for rid in common_ids if claude_labels[rid] == majority[rid]
    ) / len(common_ids)

    lines: list[str] = []
    lines.append("# Inter-Rater Reliability Report — Privacy Classification\n")
    lines.append(f"- Common items (labeled by all 3 raters): **{len(common_ids)}**")
    lines.append("- Raters: Claude (Anthropic), GPT (OpenAI, via Copilot), Gemini (Google)")
    lines.append(f"- Classes: {', '.join(PRIVACY_CLASSES)}\n")

    lines.append("## Headline numbers\n")
    lines.append(
        "Privacy classes are ordinal (public < internal < confidential). "
        "Quadratic-weighted κ is the more appropriate metric — it penalizes "
        "distant disagreements (public↔confidential) more than adjacent ones "
        "(public↔internal). Unweighted κ reported for reference.\n"
    )
    lines.append("| Metric | Value | Interpretation |")
    lines.append("|---|---|---|")
    lines.append(f"| **Fleiss' κ (3 raters, unweighted)** | **{kappa_fleiss:.3f}** | {kappa_interpretation(kappa_fleiss)} |")
    lines.append(f"| **Weighted Cohen's κ — Claude vs GPT** | **{wkappa_cg:.3f}** | {kappa_interpretation(wkappa_cg)} |")
    lines.append(f"| **Weighted Cohen's κ — Claude vs Gemini** | **{wkappa_cm:.3f}** | {kappa_interpretation(wkappa_cm)} |")
    lines.append(f"| **Weighted Cohen's κ — GPT vs Gemini** | **{wkappa_gm:.3f}** | {kappa_interpretation(wkappa_gm)} |")
    lines.append(f"| Cohen's κ (unweighted) — Claude vs GPT | {kappa_cg:.3f} | {kappa_interpretation(kappa_cg)} |")
    lines.append(f"| Cohen's κ (unweighted) — Claude vs Gemini | {kappa_cm:.3f} | {kappa_interpretation(kappa_cm)} |")
    lines.append(f"| Cohen's κ (unweighted) — GPT vs Gemini | {kappa_gm:.3f} | {kappa_interpretation(kappa_gm)} |")
    lines.append(f"| Observed agreement — all three | {all_three:.1%} | — |")
    lines.append(f"| Observed agreement — Claude/GPT | {observed_c_g:.1%} | — |")
    lines.append(f"| Observed agreement — Claude/Gemini | {observed_c_m:.1%} | — |")
    lines.append(f"| Observed agreement — GPT/Gemini | {observed_g_m:.1%} | — |")
    lines.append(f"| Claude vs. majority-vote | {claude_vs_majority:.1%} | Claude agrees with the majority on this share of items |\n")

    lines.append("## Disagreement breakdown\n")
    n_dis = len(disagreements)
    n_3way = sum(1 for d in disagreements if d["pattern"] == "3-way")
    n_2_1 = n_dis - n_3way
    lines.append(f"- Total disagreements: **{n_dis}** of {len(common_ids)} ({n_dis / len(common_ids):.1%})")
    lines.append(f"  - 2-1 split (one rater disagrees): {n_2_1}")
    lines.append(f"  - 3-way split (all three different — only possible with 3+ classes): {n_3way}\n")

    lines.append("## Confusion matrices\n")
    lines.append("### Claude (rows) × GPT (cols)\n")
    lines.append(format_matrix(cm_cg, "Claude", "GPT") + "\n")
    lines.append("### Claude (rows) × Gemini (cols)\n")
    lines.append(format_matrix(cm_cm, "Claude", "Gemini") + "\n")
    lines.append("### GPT (rows) × Gemini (cols)\n")
    lines.append(format_matrix(cm_gm, "GPT", "Gemini") + "\n")

    lines.append("## Audit-case verdict (structural misses)\n")
    lines.append(
        "The 3 cases below are confidentials that the Presidio+encoder cheap-stack "
        "structurally cannot detect (no PERSON, no high-trust entity, no encoder signal). "
        "The advisor's Option-2 claim is that Tier-5 (LLM) catches these. Here we test that "
        "claim empirically by checking what independent LLM labelers say.\n"
    )
    lines.append("| Case | id | Claude | GPT | Gemini |")
    lines.append("|---|---|---|---|---|")
    for rid, desc in AUDIT_CASES.items():
        c_l = claude_labels.get(rid, "—")
        g_l = gpt_labels.get(rid, "—")
        m_l = gemini_labels.get(rid, "—")
        lines.append(f"| {desc} | `{rid[:18]}...` | {c_l} | {g_l} | {m_l} |")
    lines.append("")

    lines.append("## Disagreement rows (for adjudication)\n")
    if not disagreements:
        lines.append("_No disagreements — all three raters agreed on every item._\n")
    else:
        lines.append("| id | Claude | GPT | Gemini | Pattern | Audit case |")
        lines.append("|---|---|---|---|---|---|")
        for d in disagreements:
            audit = d.get("audit_case") or ""
            lines.append(
                f"| `{d['id'][:18]}...` | {d['claude']} | {d['gpt']} | {d['gemini']} | {d['pattern']} | {audit} |"
            )
        lines.append("")

    lines.append("## Adjudicated majority labels\n")
    n_ambig = sum(1 for v in majority.values() if v == "AMBIG")
    lines.append(f"- Simple majority resolved: {len(common_ids) - n_ambig} / {len(common_ids)}")
    lines.append(f"- Requires human adjudication (3-way split): {n_ambig}\n")

    n_asym_confidential = sum(1 for v in asym_majority.values() if v == "confidential")
    n_claude_confidential = sum(1 for rid in common_ids if claude_labels[rid] == "confidential")
    lines.append(
        f"Under Tidus's asymmetric-safety rule "
        f"(any rater says confidential → confidential), the adjudicated confidential "
        f"count would be **{n_asym_confidential}** "
        f"(vs Claude-alone **{n_claude_confidential}**, delta = {n_asym_confidential - n_claude_confidential:+d})."
    )
    lines.append(
        "This is the count to use for the published gate analysis — it treats "
        "labeling-disagreement on confidentials as a signal to err safe.\n"
    )

    report_text = "\n".join(lines) + "\n"
    report_path = IRR_DIR / "irr_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nWrote {report_path}")

    # Emit label_overrides_irr.jsonl for rows where Claude said non-confidential
    # but at least one other rater said confidential (asymmetric-safety principle).
    # This set is applied by train_encoder._load_overrides in addition to the
    # manual label_overrides.jsonl, so the ensemble rerun sees cross-family
    # consensus-derived ground truth.
    irr_override_path = ROOT / "tests" / "classification" / "label_overrides_irr.jsonl"
    irr_flip_lines: list[str] = []
    for rid in common_ids:
        claude_l = claude_labels[rid]
        gpt_l = gpt_labels[rid]
        gem_l = gemini_labels[rid]
        if claude_l != "confidential" and (gpt_l == "confidential" or gem_l == "confidential"):
            others = [x for x in (gpt_l, gem_l) if x == "confidential"]
            irr_flip_lines.append(json.dumps({
                "id": rid,
                "privacy": "confidential",
                "reason": f"IRR cross-family adjudication: Claude={claude_l}, "
                          f"GPT={gpt_l}, Gemini={gem_l}; "
                          f"{len(others)}/2 non-Claude raters flagged confidential → "
                          "asymmetric-safety flip.",
            }))
    irr_override_path.write_text("\n".join(irr_flip_lines) + ("\n" if irr_flip_lines else ""), encoding="utf-8")
    print(f"Wrote {irr_override_path} ({len(irr_flip_lines)} asymmetric-safety flips)")

    print("\n==== Headline ====")
    print(f"Fleiss' κ (unweighted) = {kappa_fleiss:.3f}  ({kappa_interpretation(kappa_fleiss)})")
    print(f"Weighted κ  Claude-GPT    = {wkappa_cg:.3f}  ({kappa_interpretation(wkappa_cg)})")
    print(f"Weighted κ  Claude-Gemini = {wkappa_cm:.3f}  ({kappa_interpretation(wkappa_cm)})")
    print(f"Weighted κ  GPT-Gemini    = {wkappa_gm:.3f}  ({kappa_interpretation(wkappa_gm)})")
    print(f"Disagreements: {n_dis}/{len(common_ids)}")
    print(f"IRR-adjudicated confidential flips: {len(irr_flip_lines)}")


if __name__ == "__main__":
    main()
