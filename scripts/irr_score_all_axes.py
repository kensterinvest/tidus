#!/usr/bin/env python3
"""Compute IRR (weighted κ, Fleiss' κ, observed agreement) for ALL THREE
classification axes: domain, complexity, privacy.

privacy  — ordinal (public < internal < confidential) → weighted κ appropriate
complexity — ordinal (simple < moderate < complex < critical) → weighted κ appropriate
domain — nominal (chat, code, reasoning, extraction, classification,
         summarization, creative) → weighted κ NOT appropriate (no ordering);
         report unweighted Cohen/Fleiss + observed agreement only.

Reads response files from tests/classification/irr/responses/ and truth_keys.jsonl.

Run:  uv run python scripts/irr_score_all_axes.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IRR_DIR = ROOT / "tests" / "classification" / "irr"
RESP_DIR = IRR_DIR / "responses"

AXES = {
    "privacy":    {"classes": ["public", "internal", "confidential"],       "ordinal": True},
    "complexity": {"classes": ["simple", "moderate", "complex", "critical"], "ordinal": True},
    "domain":     {"classes": ["chat", "code", "reasoning", "extraction",
                               "classification", "summarization", "creative"], "ordinal": False},
}


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", line)
            if m:
                try:
                    out.append(json.loads(m.group(0)))
                except json.JSONDecodeError:
                    pass
    return out


def load_labeler(patterns: list[str], axis: str, valid: list[str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(RESP_DIR.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            for row in load_jsonl(path):
                rid = row.get("id")
                val = row.get(axis)
                if rid and val in valid:
                    merged[rid] = val
    return merged


def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    agree = sum(1 for x, y in zip(a, b) if x == y)
    p_o = agree / n
    counts_a: dict[str, int] = defaultdict(int)
    counts_b: dict[str, int] = defaultdict(int)
    for x, y in zip(a, b):
        counts_a[x] += 1
        counts_b[y] += 1
    p_e = sum((counts_a[c] / n) * (counts_b[c] / n) for c in set(counts_a) | set(counts_b))
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def weighted_kappa(a: list[str], b: list[str], classes: list[str]) -> float:
    order = {c: i for i, c in enumerate(classes)}
    n = len(a)
    if n == 0:
        return 0.0
    k = len(classes)
    w = [[((order[x] - order[y]) / (k - 1)) ** 2 for y in classes] for x in classes]
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[order[x]][order[y]] += 1
    for i in range(k):
        for j in range(k):
            obs[i][j] /= n
    rows = [sum(r) for r in obs]
    cols = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    exp = [[rows[i] * cols[j] for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * exp[i][j] for i in range(k) for j in range(k))
    if den == 0:
        return 1.0
    return 1 - (num / den)


def fleiss_kappa(raters: list[list[str]]) -> float:
    k_raters = len(raters)
    n = len(raters[0])
    classes = sorted({c for r in raters for c in r})
    per_item: list[float] = []
    totals: dict[str, int] = defaultdict(int)
    for i in range(n):
        counts: dict[str, int] = defaultdict(int)
        for r in range(k_raters):
            counts[raters[r][i]] += 1
            totals[raters[r][i]] += 1
        p_i = (sum(c * c for c in counts.values()) - k_raters) / (k_raters * (k_raters - 1))
        per_item.append(p_i)
    p_bar = sum(per_item) / n
    tot = n * k_raters
    p_e = sum((totals[c] / tot) ** 2 for c in classes)
    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def interp(k: float) -> str:
    if k < 0:
        return "poor"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "near-perfect"


def main() -> None:
    truth_rows = load_jsonl(IRR_DIR / "truth_keys.jsonl")

    print(f"{'axis':<12} {'pair':<16} {'weighted κ':>11} {'unweighted κ':>13} {'obs. agree':>10} {'interp (weighted)':>20}")
    print("-" * 100)

    for axis, conf in AXES.items():
        classes = conf["classes"]
        claude = {r["id"]: r[axis] for r in truth_rows if r.get(axis) in classes}
        gpt = load_labeler(
            ["pack_*_gpt.jsonl", "pack_*_gpt.jsonl.md", "pack_*_gpt.md"],
            axis, classes,
        )
        gem = load_labeler(
            ["pack_*_gemini.jsonl", "pack_*_gemini.jsonl.md", "pack_*_gemini.md",
             "pack_*_genmini.jsonl", "pack_*_genmini.jsonl.md", "pack_*_genmini.md"],
            axis, classes,
        )
        common = sorted(set(claude) & set(gpt) & set(gem))
        if not common:
            print(f"{axis:<12} no common ids")
            continue
        c = [claude[i] for i in common]
        g = [gpt[i] for i in common]
        m = [gem[i] for i in common]

        for pair_name, la, lb in [("Claude-GPT", c, g), ("Claude-Gemini", c, m), ("GPT-Gemini", g, m)]:
            uk = cohens_kappa(la, lb)
            obs = sum(1 for x, y in zip(la, lb) if x == y) / len(common)
            if conf["ordinal"]:
                wk = weighted_kappa(la, lb, classes)
                print(f"{axis:<12} {pair_name:<16} {wk:>11.3f} {uk:>13.3f} {obs:>10.1%} {interp(wk):>20}")
            else:
                print(f"{axis:<12} {pair_name:<16} {'n/a':>11} {uk:>13.3f} {obs:>10.1%} {interp(uk) + ' (unw)':>20}")

        fleiss = fleiss_kappa([c, g, m])
        all_three = sum(1 for x, y, z in zip(c, g, m) if x == y == z) / len(common)
        print(f"{axis:<12} {'Fleiss 3-rater':<16} {'n/a':>11} {fleiss:>13.3f} {all_three:>10.1%} {interp(fleiss) + ' (fleiss)':>20}")
        print()


if __name__ == "__main__":
    main()
