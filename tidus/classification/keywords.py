"""Tier 1 keyword matching — topic/domain hints driving complexity vetoes.

Plan.md calls for Aho-Corasick for O(n) matching on thousands of keywords.
This initial implementation uses a simple compiled regex with word boundaries —
sufficient for the ~50 keywords currently active. A.2 upgrades to pyahocorasick
when the lists expand to MeSH / legal / financial glossaries (~10k terms).

Design:
    * `match(text)` returns a list of canonical keyword IDs (lowercase, no
      punctuation), not raw matched substrings — callers log IDs.
    * Medical-adjacent keywords force `complexity=critical` (diagnosis/symptom
      contexts have real-world consequences if routed to a weak model).
    * Legal and financial keywords raise the complexity floor to `complex`
      (non-negotiable) — but don't immediately flag privacy. Presidio + T5
      handle topic-bearing confidentials.
"""
from __future__ import annotations

import re
from typing import Literal

KeywordCategory = Literal[
    "medical", "legal", "financial", "hr",
    "hardship", "credential_request",
]

# Small curated lists. Keep them conservative — false positives here force
# unnecessary complexity escalation. Expansion gated by P3 rubric review.
_KEYWORDS: dict[KeywordCategory, tuple[str, ...]] = {
    "medical": (
        "patient", "diagnose", "diagnosis", "symptom", "symptoms",
        "prescription", "HIPAA", "medical history", "ICD-10",
        "mental health", "depression", "anxiety",
    ),
    "legal": (
        "attorney", "privilege", "NDA", "non-disclosure",
        "settlement", "litigation", "subpoena", "deposition",
        "work permit", "visa status", "immigration",
    ),
    "financial": (
        "wire transfer", "tax return", "W-2", "1099",
        "earnings", "SSN", "social security", "routing number",
        "account balance", "credit score",
    ),
    "hr": (
        "employment", "employee complaint", "layoff", "terminated",
        "performance improvement plan", "PIP", "harassment",
        "discrimination", "hostile work environment",
    ),
    # Hardship — financial-distress disclosures that often lack PII entities
    # but are topic-bearing confidentials (findings.md §3).
    "hardship": (
        "no money", "cannot afford", "can't afford", "eviction",
        "homeless", "unemployed", "food stamps", "welfare",
        "debt collector", "bankruptcy", "foreclosure",
        "how do I survive", "struggling financially",
    ),
    # Credential requests — asking the model to generate or reveal secrets.
    # These are a T5 trigger because phrasing is topic-based, not entity-based.
    "credential_request": (
        "generate api key", "generate valid api",
        "give me the password", "leak the", "reveal the secret",
        "bypass authentication", "crack the", "default credentials",
    ),
}

# Compile once: categoryname -> regex that OR's all keywords in that category
_CAT_REGEX: dict[KeywordCategory, re.Pattern[str]] = {
    cat: re.compile(
        r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b",
        re.IGNORECASE,
    )
    for cat, kws in _KEYWORDS.items()
}


def match(text: str) -> dict[KeywordCategory, list[str]]:
    """Return per-category matches found in `text`. IDs are lowercased."""
    hits: dict[KeywordCategory, list[str]] = {}
    for cat, pat in _CAT_REGEX.items():
        matches = {m.group(0).lower() for m in pat.finditer(text)}
        if matches:
            hits[cat] = sorted(matches)
    return hits


def flatten(hits: dict[KeywordCategory, list[str]]) -> list[str]:
    """Flat `category:keyword` list for logging/telemetry."""
    return [f"{cat}:{kw}" for cat, kws in hits.items() for kw in kws]


def complexity_veto(hits: dict[KeywordCategory, list[str]]) -> str | None:
    """Return the complexity *floor* this category-hit set forces, or None.

    * Medical keywords       -> critical
    * Legal / financial / HR -> complex

    The classifier enforces the floor — a match can only raise complexity,
    never lower it.
    """
    if hits.get("medical"):
        return "critical"
    if any(hits.get(c) for c in ("legal", "financial", "hr")):
        return "complex"
    return None
