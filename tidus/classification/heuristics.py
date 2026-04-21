"""Tier 1 heuristic fast-path — regex/PII/secret detection + structural signals.

Design:
    * Every regex has a stable pattern ID (e.g., "SSN", "AWS_ACCESS_KEY").
      Pattern IDs surface in logs; match values NEVER do (plan.md telemetry rule).
    * Credit-card hits are validated with Luhn before being counted — raw 16-digit
      strings include many non-card identifiers.
    * Short-circuit decision:
        - Any PII/secret hit       -> privacy=confidential (asymmetric safety)
        - Code fence / code syntax -> domain=code hint (encoder still decides)
        - Nothing found            -> fall through to Tier 2

The regex set mirrors scripts/label_wildchat.py (the rubric the encoder was
trained against) but adds Luhn + named pattern IDs. Keyword matching lives
in `keywords.py` to keep this file focused on character-level patterns.
"""
from __future__ import annotations

import re

from tidus.classification.models import Tier1Signals

# Pattern ID → (compiled regex, needs_luhn)
#
# CREDIT_CARD constraints: first digit must match a known BIN prefix range
# (3 = Amex / Diners, 4 = Visa, 5 = MC, 6 = Discover) AND the full string
# must pass Luhn. Both gates together suppress arbitrary Luhn-valid digit
# runs (serial numbers, MD5 fragments, hash collisions) while keeping any
# real card match.
#
# AWS_SECRET_KEY (40-char base64) is intentionally absent — it matches any
# base64-encoded blob of the right length (images, hashes, tokens). The
# detect-secrets wiring in A.1.5 will pick it up with entropy + context.
_REGEX_PATTERNS: dict[str, tuple[re.Pattern[str], bool]] = {
    "SSN_US":            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), False),
    "CREDIT_CARD":       (re.compile(r"\b[3-6](?:[ -]?\d){12,18}\b"), True),
    "EMAIL":             (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), False),
    "PHONE_INTL":        (re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d"), False),
    "AWS_ACCESS_KEY":    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), False),
    "GITHUB_TOKEN":      (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), False),
    "GITHUB_PAT":        (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{80,}\b"), False),
    "OPENAI_KEY":        (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), False),
    "ANTHROPIC_KEY":     (re.compile(r"\bsk-ant-[A-Za-z0-9-]{90,}\b"), False),
    "SLACK_TOKEN":       (re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b"), False),
    "JWT":               (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), False),
    "PRIVATE_KEY_HEADER": (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |)PRIVATE KEY-----"), False),
}

# Regex patterns that trigger privacy=confidential immediately (asymmetric-safety).
# Non-privacy patterns (e.g., EMAIL alone) are signals but not definitive.
_CONFIDENTIAL_PATTERN_IDS = frozenset({
    "SSN_US",
    "CREDIT_CARD",
    "AWS_ACCESS_KEY",
    "GITHUB_TOKEN",
    "GITHUB_PAT",
    "OPENAI_KEY",
    "ANTHROPIC_KEY",
    "SLACK_TOKEN",
    "PRIVATE_KEY_HEADER",
    "JWT",
    # EMAIL and PHONE are intentionally NOT here — a message that just mentions
    # "contact@example.com" as a placeholder isn't automatically confidential.
    # Presidio + encoder confirm those downstream.
})

# Code-structure detection — hint for domain=code but not definitive.
_CODE_FENCE_RE = re.compile(r"```|^\s*(def |class |import |from |\$\s)", re.M)

# Rough token estimator — 4 chars/token average for English is the standard
# GPT-tokenizer heuristic. Good enough for routing; exact counts come from
# the vendor tokenizer at request time.
_CHARS_PER_TOKEN = 4


def _luhn_valid(digits: str) -> bool:
    """Luhn mod-10 check on a digit-only string. Rejects lengths outside 13..19."""
    digits = re.sub(r"\D", "", digits)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _find_regex_hits(text: str) -> list[str]:
    """Return pattern IDs that fired. Luhn-gated patterns are validated."""
    hits: list[str] = []
    for pid, (pat, needs_luhn) in _REGEX_PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        if needs_luhn and not _luhn_valid(m.group(0)):
            continue
        hits.append(pid)
    return hits


def estimate_tokens(text: str) -> int:
    """Char-based token estimate. Matches the 4 chars/token industry heuristic."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def run_tier1(text: str, keyword_hits: list[str] | None = None) -> Tier1Signals:
    """Evaluate Tier 1 heuristics against `text`.

    `keyword_hits` is supplied by the caller from `keywords.match()` — kept
    as a parameter so this function stays pure and testable in isolation.
    """
    regex_hits = _find_regex_hits(text)
    kw = list(keyword_hits or [])
    return Tier1Signals(
        regex_hits=regex_hits,
        secret_types=[],  # detect-secrets wiring deferred to A.1.5
        keyword_hits=kw,
        has_code_fence=bool(_CODE_FENCE_RE.search(text)),
        estimated_input_tokens=estimate_tokens(text),
        any_hit=bool(regex_hits or kw),
    )


def any_confidential_regex(signals: Tier1Signals) -> bool:
    """True iff any of signals.regex_hits forces privacy=confidential.

    Asymmetric-safety OR: a single definitive hit (SSN, API key, etc.)
    is enough — we don't require multiple signals to agree.
    """
    return any(pid in _CONFIDENTIAL_PATTERN_IDS for pid in signals.regex_hits)
