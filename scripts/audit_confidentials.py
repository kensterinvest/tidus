#!/usr/bin/env python3
"""Audit all `privacy=confidential` labels — bucket by heuristic flags for review.

The POC backtest eyeball check revealed that labels_001-045 contain likely
labeler overcalls where fictional-named-character medical narratives were
marked confidential. Per taxonomy, fictional characters are `public`.
This script surfaces each confidential with text + flags so we can decide
which stay (genuine PII/credentials/real-names-with-context) and which flip.

Output: structured printout grouped by flag combo, written to stdout and
an optional --out file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHUNKS = REPO / "tests" / "classification" / "chunks"
POOL = REPO / "tests" / "classification" / "pool_chunks"

FICTION_CHARS = [
    # Recurring medical-narrative fictional characters observed in earlier chunks
    "Britt Lindqvist", "Reza Fazekas", "Loretta Miller", "Susan Schmidt",
    "Patricia Bertier", "Caroline Hertig", "Alienor Cuillerier",
    "Jewel Whitehead", "Anneliese Ehn", "Huguette Weissbrodt",
    "Beatrix Tell", "Sabrine", "Luciana Vipari", "Mysterik", "Hertig",
    "Marcel Messier", "Flamand", "Pascal Flamand", "Albert Hertig",
    "Mireille", "Basil Bends", "Percival Gleswick", "Jacques-Jean Morel",
    "Otto Loosli", "Wiebke Wittich",
]

MEDFETISH_KEYWORDS = [
    "paralysis", "wheelchair", "orthopedic brace", "adult diaper",
    "hip spica cast", "pelvic brace", "incontinence", "tracheostomy",
    "bedridden", "fractured pelvis", "immobilized",
]

# Patterns for GENUINE credentials/secrets (keep as confidential)
CREDENTIAL_PATTERNS = [
    (r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9+/_\-=]{20,}", "GENERIC_SECRET"),
    (r"\bsk-[A-Za-z0-9]{20,}\b", "OPENAI_KEY"),
    (r"\bEAA[A-Za-z0-9]{50,}\b", "FACEBOOK_GRAPH_TOKEN"),
    (r"\bvk\d\.[a-zA-Z0-9._-]{50,}", "VK_TOKEN"),
    (r"\bAKIA[A-Z0-9]{16}\b", "AWS_KEY"),
    (r"\bgh[pousr]_[A-Za-z0-9]{36,}\b", "GH_TOKEN"),
    (r"\bBWGc[A-Za-z0-9]{40,}", "PEXELS_KEY"),
]


def flag_text(text: str) -> list[str]:
    flags: list[str] = []
    head = text[:200]
    if "Ignore previous instructions" in head:
        flags.append("FICTION_OPENER")
    for name in FICTION_CHARS:
        if name in text:
            flags.append(f"FICTION_CHAR[{name}]")
            break
    low = text.lower()
    if any(kw in low for kw in MEDFETISH_KEYWORDS):
        flags.append("MEDFETISH_TROPE")
    for pat, tag in CREDENTIAL_PATTERNS:
        if re.search(pat, text):
            flags.append(f"CRED[{tag}]")
    # Real-looking student IDs / register numbers
    if re.search(r"Register Number\s*:?\s*\d{10,}", text):
        flags.append("STUDENT_REG_ID")
    # Chinese phone number + name patterns (like 姓名：张三 电话：138...)
    if re.search(r"姓名[：:]\s*\S+.*电话[：:]\s*1[3-9]\d{9}", text):
        flags.append("CN_PHONE_PII")
    # Real-looking named customer with financial context
    if re.search(r'"customer_name"\s*:\s*"(?!string)', text) and any(k in text for k in ["loan_amount", "commission", "bank_received"]):
        flags.append("CUSTOMER_FINANCIAL_PII")
    # Presidio-redaction marker is a tell
    if "<PRESIDIO_ANONYMIZED_EMAIL_ADDRESS>" in text:
        flags.append("PRESIDIO_REDACTED")
    # Internal company package paths (Taobao / Alibaba etc.)
    if re.search(r"com\.(taobao|alibaba|simba)\.", text):
        flags.append("CN_CORP_PACKAGE")
    # MySQL hardcoded admin creds
    if re.search(r'\$password\s*=\s*"[^"]{5,}"', text) or re.search(r'password\s*=\s*"admin', text, re.I):
        flags.append("MYSQL_HARDCODED")
    return flags


def suggest(flags: list[str]) -> str:
    """Heuristic: recommend KEEP (genuine), FLIP (overcall to public), or REVIEW."""
    has_cred = any(f.startswith("CRED[") for f in flags)
    has_cred = has_cred or "STUDENT_REG_ID" in flags or "CN_PHONE_PII" in flags
    has_cred = has_cred or "CUSTOMER_FINANCIAL_PII" in flags or "MYSQL_HARDCODED" in flags
    has_internal_corp = "CN_CORP_PACKAGE" in flags or "PRESIDIO_REDACTED" in flags
    is_fiction = any(f.startswith("FICTION_") for f in flags) or "MEDFETISH_TROPE" in flags

    if has_cred:
        return "KEEP"
    if is_fiction and not has_internal_corp:
        return "FLIP"
    return "REVIEW"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help="Write report to this file")
    args = parser.parse_args()

    pool_text: dict[str, str] = {}
    for pf in sorted(POOL.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                pool_text[r["id"]] = r["text"]

    confidentials = []
    for lf in sorted(CHUNKS.glob("labels_*.jsonl")):
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["privacy"] != "confidential":
                    continue
                text = pool_text.get(r["id"], "")
                flags = flag_text(text)
                confidentials.append({
                    "chunk": lf.stem,
                    "id": r["id"],
                    "domain": r["domain"],
                    "complexity": r["complexity"],
                    "rationale": r["rationale"],
                    "text_head": text[:250],
                    "text_full_len": len(text),
                    "flags": flags,
                    "suggest": suggest(flags),
                })

    buf: list[str] = []
    def out(s: str = ""):
        buf.append(s)

    out(f"# Confidential label audit — {len(confidentials)} total")
    out()
    suggestions = defaultdict(int)
    for c in confidentials:
        suggestions[c["suggest"]] += 1
    out("Heuristic suggestion totals:")
    for k, v in suggestions.items():
        out(f"  {k}: {v}")
    out()
    out("=" * 80)

    # Group by suggestion → then by chunk for easier scanning
    for bucket in ("KEEP", "FLIP", "REVIEW"):
        items = [c for c in confidentials if c["suggest"] == bucket]
        out(f"\n## {bucket} ({len(items)})\n")
        for c in items:
            preview = " ".join(c["text_head"].split())[:200]
            out(f"- [{c['chunk']}] {c['id']}   dom={c['domain']} cmp={c['complexity']}  len={c['text_full_len']}")
            out(f"    flags: {c['flags']}")
            out(f"    rationale: {c['rationale'][:180]}")
            out(f"    text: {preview}")
            out()

    report = "\n".join(buf)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        sys.stderr.write(f"\nWrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
