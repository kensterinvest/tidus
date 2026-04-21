"""Prometheus metrics for the classification cascade.

Exposed at `/metrics` alongside the registry metrics.

Counters (2):
  tidus_classify_t5_calls_total{result}   — T5 call outcomes
  tidus_classify_t5_flips_total{from_privacy}  — T5 privacy-upgrade flips by prior verdict

Histogram (1):
  tidus_classify_t5_latency_seconds{result}  — T5 wall-clock latency per call

`result` label values for calls_total + latency_seconds:
  - "success"       — /api/chat returned a parseable label set
  - "failure"       — network error, parse error, schema mismatch
  - "cache_hit"     — TTL cache served the response (network NOT hit)
  - "rate_limited"  — sliding-window limiter rejected the request

`from_privacy` label on flips_total is the prior merged verdict before T5
fired. Flip direction is always → confidential (T5 is asymmetric-safety only).

Usage (from tidus.classification.llm_classifier):
    from tidus.observability.classification_metrics import T5_CALLS_TOTAL
    T5_CALLS_TOTAL.labels(result="cache_hit").inc()
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

T5_CALLS_TOTAL = Counter(
    "tidus_classify_t5_calls_total",
    "T5 LLM classifier call count, labelled by outcome",
    labelnames=["result"],
)

T5_FLIPS_TOTAL = Counter(
    "tidus_classify_t5_flips_total",
    "T5 privacy-upgrade flip count, labelled by the prior (pre-T5) privacy verdict",
    labelnames=["from_privacy"],
)

# Buckets tuned for the T5 p95 <= 500 ms target on Enterprise SKU; upper
# buckets kept past 5 s to catch pathological Ollama stalls.
T5_LATENCY_SECONDS = Histogram(
    "tidus_classify_t5_latency_seconds",
    "T5 LLM classifier wall-clock latency in seconds, labelled by outcome",
    labelnames=["result"],
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0),
)
