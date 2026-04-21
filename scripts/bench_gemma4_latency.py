#!/usr/bin/env python3
"""Latency gate for Gemma-4 T5-swap candidate (plan.md dual-gate protocol).

Runs N representative prompts through Ollama with JSON-format-constrained
output using the frozen SYSTEM_PROMPT from scripts/label_wildchat.py.
Measures end-to-end latency (HTTP POST → JSON emit). Reports p50/p95/p99.

Fail-fast: after --warm warmup calls, if p50 > 500 ms the model is
disqualified for Tidus's 4-vCPU no-GPU T5 deployment constraint
regardless of subsequent accuracy.

Usage:
    uv run python scripts/bench_gemma4_latency.py --model gemma4:e4b-it-q4_K_M
    uv run python scripts/bench_gemma4_latency.py --model gemma4:e4b-it-q4_K_M --n 200
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = REPO_ROOT / "tests" / "classification" / "pool_chunks"
CHUNKS_DIR = REPO_ROOT / "tests" / "classification" / "chunks"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from label_wildchat import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402

MAX_CHARS = 1200  # match encoder preprocessing
OLLAMA_URL = "http://localhost:11434/api/chat"
T5_BUDGET_MS = 500  # plan.md dual-gate latency ceiling


def load_unlabeled(sample_n: int, seed: int) -> list[tuple[str, str]]:
    labeled: set[str] = set()
    for lf in sorted(CHUNKS_DIR.glob("labels_*.jsonl")):
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                labeled.add(json.loads(line)["id"])

    pool: list[tuple[str, str]] = []
    for pf in sorted(POOL_DIR.glob("pool_*.jsonl")):
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["id"] not in labeled:
                    pool.append((r["id"], r["text"][:MAX_CHARS]))
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:sample_n]


def classify_once(client: httpx.Client, model: str, text: str, num_predict: int) -> tuple[float, str, int]:
    """Returns (elapsed_ms, raw_response_json, eval_tokens)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(message=text)},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": num_predict,
            "temperature": 0.0,  # deterministic for latency consistency
        },
    }
    start = time.perf_counter()
    resp = client.post(OLLAMA_URL, json=payload, timeout=120.0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    resp.raise_for_status()
    body = resp.json()
    content = body.get("message", {}).get("content", "")
    eval_count = int(body.get("eval_count", 0))
    return elapsed_ms, content, eval_count


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    xs_sorted = sorted(xs)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=200, help="prompt count")
    parser.add_argument("--warm", type=int, default=3, help="warmup prompts excluded from stats")
    parser.add_argument("--num-predict", type=int, default=120, help="max tokens per response")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-fast-after", type=int, default=30,
                        help="after this many non-warmup calls, abort if p50 > budget")
    parser.add_argument("--budget-ms", type=int, default=T5_BUDGET_MS)
    args = parser.parse_args()

    prompts = load_unlabeled(args.n + args.warm, args.seed)
    if len(prompts) < args.n + args.warm:
        print(f"WARN: only {len(prompts)} unlabeled prompts available")
    print(f"Loaded {len(prompts)} prompts ({args.warm} warmup + {args.n} measured)")
    print(f"Model: {args.model}    Budget: {args.budget_ms} ms")
    print(f"Fail-fast: after {args.fail_fast_after} non-warmup, if p50 > budget")

    latencies: list[float] = []
    token_counts: list[int] = []
    parse_errors = 0

    with httpx.Client() as client:
        for i, (pid, text) in enumerate(prompts):
            tag = "WARM" if i < args.warm else f"{i - args.warm + 1:>3}/{args.n}"
            try:
                lat, content, eval_tok = classify_once(client, args.model, text, args.num_predict)
            except Exception as exc:
                print(f"[{tag}] ERROR: {exc}")
                return 2

            parse_ok = True
            try:
                json.loads(content)
            except Exception:
                parse_ok = False

            if i >= args.warm:
                latencies.append(lat)
                token_counts.append(eval_tok)
                if not parse_ok:
                    parse_errors += 1

            marker = "OK " if parse_ok else "BAD"
            print(f"[{tag}] {lat:>7.1f} ms  tokens={eval_tok:>3}  {marker}  id={pid[-12:]}")

            if i >= args.warm + args.fail_fast_after:
                p50 = percentile(latencies, 0.50)
                if p50 > args.budget_ms:
                    print(f"\nFAIL-FAST: after {len(latencies)} samples, p50={p50:.1f} ms > budget={args.budget_ms} ms")
                    break

    if not latencies:
        print("No latencies recorded.")
        return 1

    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)
    p99 = percentile(latencies, 0.99)
    mean = statistics.mean(latencies)
    mean_tok = statistics.mean(token_counts) if token_counts else 0

    print("\n" + "=" * 60)
    print(f"N = {len(latencies)} (after {args.warm} warmup)")
    print(f"Mean latency:   {mean:>7.1f} ms")
    print(f"p50:            {p50:>7.1f} ms")
    print(f"p95:            {p95:>7.1f} ms")
    print(f"p99:            {p99:>7.1f} ms")
    print(f"Mean tokens:    {mean_tok:>7.1f}")
    if mean_tok > 0:
        print(f"Tokens/sec:     {mean_tok / (mean / 1000):>7.2f}")
    print(f"Parse errors:   {parse_errors}/{len(latencies)}")
    print("=" * 60)

    if p95 <= args.budget_ms:
        print(f"\nPASS: p95 {p95:.1f} ms <= budget {args.budget_ms} ms")
        return 0
    print(f"\nFAIL: p95 {p95:.1f} ms > budget {args.budget_ms} ms")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
