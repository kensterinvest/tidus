#!/usr/bin/env python3
"""Tidus v1.3.0 -- Phase 0 Step 1: WildChat-1M Sonnet Labeler.

Samples 1000 stratified prompts from Allen AI's WildChat-1M (real ChatGPT/Claude
conversations with real PII patterns), labels each via Claude Sonnet 4.6, saves
to `tests/classification/real_traffic_eval.jsonl` as the permanent eval harness.

This is the Phase 0 GATE dataset — every future v1.3 classifier change is
measured against it. Do NOT regenerate casually; freeze after first run.

Usage:
    uv run python scripts/label_wildchat.py --dry-run           # 10 samples, no API calls
    uv run python scripts/label_wildchat.py --sample 10         # 10 real labels (~$0.03)
    uv run python scripts/label_wildchat.py                     # 1000 full run (~$3)

Data source:
    allenai/WildChat-1M (ODC-BY, Hugging Face)
    License compatible with commercial eval use; derivative artifacts (confusion
    matrices) are redistributable without carve-out.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from anthropic import AsyncAnthropic
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()  # loads ANTHROPIC_API_KEY from .env in project root

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"  # used only in --api-label mode
OUTPUT_PATH = Path("tests/classification/real_traffic_eval.jsonl")
POOL_PATH = Path("tests/classification/prompts_pool.jsonl")
TARGET_SAMPLES = 5000
POOL_SIZE = 25_000  # stream this many WildChat conversations to stratify from
CONCURRENCY = 10  # parallel Sonnet API calls (--api-label mode only)
MAX_INPUT_CHARS = 2000  # truncation per Tidus v1.3 input policy
SEED = 42

# Cost per 1M tokens (Sonnet 4.6, non-batch)
PRICE_IN_PER_M = 3.0
PRICE_OUT_PER_M = 15.0

# ---------------------------------------------------------------------------
# Enums (mirror tidus/models/task.py — keep in sync)
# ---------------------------------------------------------------------------

class Domain(str, Enum):
    chat = "chat"
    code = "code"
    reasoning = "reasoning"
    extraction = "extraction"
    classification = "classification"
    summarization = "summarization"
    creative = "creative"


class Complexity(str, Enum):
    simple = "simple"
    moderate = "moderate"
    complex = "complex"
    critical = "critical"


class Privacy(str, Enum):
    public = "public"
    internal = "internal"
    confidential = "confidential"


# ---------------------------------------------------------------------------
# Labeling prompt template
# ---------------------------------------------------------------------------
# DESIGN NOTE — KENNY: This is the most important design choice in Phase 0.
# The prompt defines what the encoder will learn. Three core decisions below:
#
#   (1) Taxonomy definitions — kept tight and unambiguous
#   (2) Privacy asymmetry — "when uncertain → confidential" matches Tidus's
#       compliance stance; if removed, encoder will be too liberal on privacy
#   (3) Rationale field — costs ~40 output tokens per label but enables human
#       spot-checking during clean-eval-tier verification; recommended to keep
#
# Alternatives to consider:
#   - Add 2-3 few-shot examples (improves consistency, costs ~200 input tokens
#     per call, ~$0.60 added to batch). Recommended if initial labeled sample
#     shows Sonnet inconsistency.
#   - Force chain-of-thought before JSON (higher quality, slower, more expensive).
#     Probably overkill for 14-class routing classification.
#
# Modify below if you want to tweak before the full $3 run.

SYSTEM_PROMPT = """\
You are an expert classifier for an enterprise AI routing system. You classify \
each user message across three dimensions. You output ONLY a single JSON object \
with fields: domain, complexity, privacy, rationale. No preamble, no trailing \
commentary.

TAXONOMY

domain — what TYPE of task (not topic):
  chat            = conversational, open-ended, no clear deliverable
  code            = writing/debugging/explaining code; shell commands; SQL
  reasoning       = logic, math proofs, step-by-step analysis, planning
  extraction      = pulling structured data from unstructured input
  classification  = assigning labels/categories to input
  summarization   = condensing longer input into shorter output
  creative        = fiction, poetry, brainstorming, roleplay, marketing copy

complexity — cognitive load needed for a correct answer:
  simple    = one-step lookup/answer; trivially verifiable
  moderate  = multi-step but bounded scope
  complex   = architecture, system design, advanced reasoning
  critical  = medical diagnosis, legal advice, financial planning, compliance —
              wrong answer has material real-world consequences

privacy — sensitivity of the content itself:
  public        = no sensitive info; could be posted on a public forum
  internal      = business content, work tasks, routine questions
  confidential  = contains PII (SSN, credit cards, real names+context),
                  secrets (API keys, passwords, tokens),
                  medical/legal/financial specifics tied to a person or org

RULES

- Classify the REQUEST the user is making, not the topic they mention.
  "Summarize this Python tutorial" → domain=summarization, not code.
- When privacy is ambiguous between internal and confidential, choose
  confidential. False negatives here are compliance incidents; overclassification
  is safe. This is asymmetric cost.
- Never output privacy=public if the message contains real names, addresses,
  phone numbers, emails, account numbers, or any identifier.
- rationale = ONE sentence explaining the domain+complexity choice.

OUTPUT FORMAT (valid JSON, nothing else)

{"domain": "...", "complexity": "...", "privacy": "...", "rationale": "..."}
"""

USER_PROMPT_TEMPLATE = """\
Classify this message:

<<<
{message}
>>>

Respond with JSON only."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LabelResult:
    id: str
    text: str
    text_length: int
    has_code_fence: bool
    has_pii_pattern: bool
    domain: str | None
    complexity: str | None
    privacy: str | None
    rationale: str | None
    tier: str = "sonnet"  # for compat with clean-eval-tier downstream
    parse_error: str | None = None


# ---------------------------------------------------------------------------
# Stratified sampling from WildChat stream
# ---------------------------------------------------------------------------

CODE_FENCE_RE = re.compile(r"```|^\s*(def |class |import |from |\$\s)", re.M)
PII_HINT_RE = re.compile(
    r"\b(\d{3}-\d{2}-\d{4}|\d{16}|\d{4}-\d{4}-\d{4}-\d{4}|"  # SSN, CC
    r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|"                  # AWS, GH
    r"sk-[A-Za-z0-9]{20,}|"                                   # OpenAI-style
    r"\b\w+@\w+\.\w+|\+?\d[\d\s\-\(\)]{7,}\d)\b"              # email, phone
)
KEYWORD_HINT_RE = re.compile(
    r"\b(patient|diagnos\w*|symptom|HIPAA|attorney|privilege|NDA|wire transfer|"
    r"tax return|W-2|earnings|SSN|social security)\b",
    re.IGNORECASE,
)


def extract_first_user_turn(conv: list[dict]) -> str | None:
    """Return the first user message content from a WildChat conversation."""
    for msg in conv:
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if content:
                return content[:MAX_INPUT_CHARS]
    return None


def is_english(row: dict) -> bool:
    """Filter non-English conversations per Tidus v1.3 multilingual policy."""
    return row.get("language", "").lower() == "english"


def stratify(pool: list[dict], target: int, seed: int = SEED) -> list[dict]:
    """Sample `target` prompts from `pool` with stratified boost for code/PII/keyword hits.

    Target composition (approximate):
      - 15% prompts with code fences  → code-domain representation
      - 12% prompts with PII patterns → confidential stress
      - 10% prompts with medical/legal/financial keywords
      - 63% random baseline (drawn from remainder)
    """
    rng = random.Random(seed)

    code_bucket = [p for p in pool if CODE_FENCE_RE.search(p["text"])]
    pii_bucket = [p for p in pool if PII_HINT_RE.search(p["text"])]
    kw_bucket = [p for p in pool if KEYWORD_HINT_RE.search(p["text"])]
    baseline = [p for p in pool if p not in code_bucket and p not in pii_bucket and p not in kw_bucket]

    sample: list[dict] = []
    sample += rng.sample(code_bucket, min(int(target * 0.15), len(code_bucket)))
    sample += rng.sample(pii_bucket, min(int(target * 0.12), len(pii_bucket)))
    sample += rng.sample(kw_bucket, min(int(target * 0.10), len(kw_bucket)))
    sample += rng.sample(baseline, min(target - len(sample), len(baseline)))

    rng.shuffle(sample)
    return sample[:target]


def build_pool(pool_size: int) -> list[dict]:
    """Stream WildChat-1M and collect `pool_size` candidate English prompts."""
    print(f"[pool] Streaming allenai/WildChat-1M (target pool={pool_size})...", file=sys.stderr)
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    pool: list[dict] = []
    for row in ds:
        if len(pool) >= pool_size:
            break
        if not is_english(row):
            continue
        text = extract_first_user_turn(row.get("conversation", []))
        if not text or len(text) < 10:
            continue
        pool.append({
            "id": f"wildchat-{row.get('conversation_hash', len(pool))}",
            "text": text,
        })
        if len(pool) % 500 == 0:
            print(f"[pool] collected {len(pool)}/{pool_size}", file=sys.stderr)
    print(f"[pool] done ({len(pool)} prompts)", file=sys.stderr)
    return pool


# ---------------------------------------------------------------------------
# Sonnet labeling
# ---------------------------------------------------------------------------

JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_sonnet_json(raw: str) -> tuple[dict | None, str | None]:
    """Parse Sonnet's JSON response. Returns (labels_dict, error)."""
    raw = raw.strip()
    if not raw.startswith("{"):
        match = JSON_RE.search(raw)
        if not match:
            return None, f"no JSON object found in: {raw[:120]}"
        raw = match.group(0)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"
    for field in ("domain", "complexity", "privacy"):
        if field not in obj:
            return None, f"missing field: {field}"
    try:
        Domain(obj["domain"])
        Complexity(obj["complexity"])
        Privacy(obj["privacy"])
    except ValueError as e:
        return None, f"invalid enum value: {e}"
    return obj, None


async def label_one(client: AsyncAnthropic, prompt: dict) -> LabelResult:
    """Label a single prompt via Sonnet."""
    user_msg = USER_PROMPT_TEMPLATE.format(message=prompt["text"])
    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text if response.content else ""
        obj, err = parse_sonnet_json(raw)
    except Exception as e:  # SDK retries transient errors internally
        obj, err = None, f"api_error: {type(e).__name__}: {e}"

    return LabelResult(
        id=prompt["id"],
        text=prompt["text"],
        text_length=len(prompt["text"]),
        has_code_fence=bool(CODE_FENCE_RE.search(prompt["text"])),
        has_pii_pattern=bool(PII_HINT_RE.search(prompt["text"])),
        domain=obj["domain"] if obj else None,
        complexity=obj["complexity"] if obj else None,
        privacy=obj["privacy"] if obj else None,
        rationale=obj.get("rationale") if obj else None,
        parse_error=err,
    )


async def label_batch(prompts: list[dict]) -> list[LabelResult]:
    """Label all prompts with bounded concurrency."""
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(p):
        async with sem:
            return await label_one(client, p)

    print(f"[label] starting {len(prompts)} requests (concurrency={CONCURRENCY})...", file=sys.stderr)
    t0 = time.time()
    results: list[LabelResult] = []
    for i, coro in enumerate(asyncio.as_completed([bounded(p) for p in prompts])):
        res = await coro
        results.append(res)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate
            print(f"[label] {i+1}/{len(prompts)}  rate={rate:.1f}/s  eta={eta:.0f}s", file=sys.stderr)
    print(f"[label] done in {time.time()-t0:.0f}s", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(results: list[LabelResult]) -> None:
    total = len(results)
    errors = [r for r in results if r.parse_error]
    ok = [r for r in results if not r.parse_error]
    print("\n=== Labeling summary ===", file=sys.stderr)
    print(f"Total: {total}", file=sys.stderr)
    print(f"OK:    {len(ok)} ({len(ok)/total:.1%})", file=sys.stderr)
    print(f"Errors:{len(errors)} ({len(errors)/total:.1%})", file=sys.stderr)
    if errors[:3]:
        print("\nFirst 3 errors:", file=sys.stderr)
        for r in errors[:3]:
            print(f"  [{r.id}] {r.parse_error}", file=sys.stderr)
    print("\nDomain distribution:", file=sys.stderr)
    for k, v in sorted(Counter(r.domain for r in ok).items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/len(ok):.1%})", file=sys.stderr)
    print("\nComplexity distribution:", file=sys.stderr)
    for k, v in sorted(Counter(r.complexity for r in ok).items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/len(ok):.1%})", file=sys.stderr)
    print("\nPrivacy distribution:", file=sys.stderr)
    for k, v in sorted(Counter(r.privacy for r in ok).items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/len(ok):.1%})", file=sys.stderr)


def estimate_cost(n: int) -> tuple[float, float, float]:
    """Rough cost estimate for n labels (system ~400 tok, avg user ~500 tok, avg out ~80 tok)."""
    in_tokens = n * (400 + 500)  # system + user
    out_tokens = n * 80
    cost_in = in_tokens / 1e6 * PRICE_IN_PER_M
    cost_out = out_tokens / 1e6 * PRICE_OUT_PER_M
    return cost_in, cost_out, cost_in + cost_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="""\
Tidus v1.3.0 Phase 0 Step 1 — WildChat prompt stratifier.

Three modes:
  --dump-only   Stream WildChat, stratify, write prompts-only JSONL (no labels).
                Use this when labeling will be done inline by Claude in-session
                (free under a Claude Max plan, higher quality than Sonnet API).
  --api-label   Label via Anthropic Sonnet API in-script (costs money, needs key).
  --dry-run     Print first 3 stratified prompts and exit (no API, no file write).
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=TARGET_SAMPLES,
                   help=f"number of prompts (default {TARGET_SAMPLES})")
    p.add_argument("--pool", type=int, default=POOL_SIZE,
                   help=f"WildChat pool size to stratify from (default {POOL_SIZE})")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dump-only", action="store_true",
                      help="Stratify and write prompts-only JSONL (no API).")
    mode.add_argument("--api-label", action="store_true",
                      help="Label via Sonnet API (costs money, needs ANTHROPIC_API_KEY).")
    mode.add_argument("--dry-run", action="store_true",
                      help="Show first 3 stratified prompts and exit.")
    p.add_argument("--output", type=Path, default=None,
                   help=f"Output path. Default: {POOL_PATH} (dump-only) or {OUTPUT_PATH} (api-label)")
    p.add_argument("--yes", action="store_true", help="Skip cost prompt (api-label only)")
    args = p.parse_args()

    # Default output depends on mode
    if args.output is None:
        args.output = POOL_PATH if args.dump_only else OUTPUT_PATH
    # Default mode = dump-only (safest, free)
    if not (args.dump_only or args.api_label or args.dry_run):
        args.dump_only = True
    return args


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Build stratified sample (shared across all modes)
    pool = build_pool(args.pool)
    sample = stratify(pool, args.sample)
    print(f"[sample] selected {len(sample)} prompts", file=sys.stderr)

    # --dry-run: show first 3 prompts, exit
    if args.dry_run:
        print("\n=== Dry run — first 3 stratified prompts ===", file=sys.stderr)
        for p in sample[:3]:
            print(f"\n[{p['id']}] ({len(p['text'])} chars)", file=sys.stderr)
            print(p["text"][:500], file=sys.stderr)
        return

    # --dump-only: write prompts to JSONL, exit (no API calls)
    if args.dump_only:
        print(f"\n[dump] Writing {len(sample)} prompts to {args.output}", file=sys.stderr)
        with args.output.open("w", encoding="utf-8") as f:
            for p in sample:
                row = {
                    "id": p["id"],
                    "text": p["text"],
                    "text_length": len(p["text"]),
                    "has_code_fence": bool(CODE_FENCE_RE.search(p["text"])),
                    "has_pii_pattern": bool(PII_HINT_RE.search(p["text"])),
                    "has_keyword_hint": bool(KEYWORD_HINT_RE.search(p["text"])),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[dump] Done. Stratum counts:", file=sys.stderr)
        code_n = sum(1 for p in sample if CODE_FENCE_RE.search(p["text"]))
        pii_n = sum(1 for p in sample if PII_HINT_RE.search(p["text"]))
        kw_n = sum(1 for p in sample if KEYWORD_HINT_RE.search(p["text"]))
        print(f"  code-fence:   {code_n} ({code_n/len(sample):.1%})", file=sys.stderr)
        print(f"  pii-pattern:  {pii_n} ({pii_n/len(sample):.1%})", file=sys.stderr)
        print(f"  keyword-hint: {kw_n} ({kw_n/len(sample):.1%})", file=sys.stderr)
        return

    # --api-label: label via Sonnet API
    cost_in, cost_out, cost_total = estimate_cost(args.sample)
    print(f"\nPlan: label {args.sample} prompts via {MODEL} (external API)", file=sys.stderr)
    print(f"Est. cost: ${cost_in:.2f} in + ${cost_out:.2f} out = ${cost_total:.2f}", file=sys.stderr)
    print(f"Output:    {args.output}", file=sys.stderr)
    if not args.yes:
        resp = input("Proceed? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborted.", file=sys.stderr)
            sys.exit(1)

    results = asyncio.run(label_batch(sample))
    with args.output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    summarize(results)
    print(f"\nWritten: {args.output} ({len(results)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
