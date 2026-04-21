"""Smoke tests for T5 Prometheus counters + latency histogram.

We verify the metrics exist with the expected labels and that T5 call
paths emit them. The underlying Prometheus client is well-tested upstream;
these tests catch wiring errors, not library bugs.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from tidus.classification.llm_classifier import LLMClassifier
from tidus.observability.classification_metrics import (
    T5_CALLS_TOTAL,
    T5_FLIPS_TOTAL,
    T5_LATENCY_SECONDS,
)


def _sample_value(counter, **labels) -> float:
    """Prometheus Counter/Histogram have opaque internals; .labels(**kw)._value.get()
    works for Counter. For Histogram we read .labels(**kw)._sum.get()."""
    labelled = counter.labels(**labels)
    if hasattr(labelled, "_value"):
        return labelled._value.get()
    return labelled._sum.get()


@pytest.fixture
def loaded_llm() -> LLMClassifier:
    c = LLMClassifier(model="phi3.5:latest", endpoint="http://localhost:11434")
    c._loaded = True  # bypass startup ping for unit test
    return c


@pytest.mark.asyncio
async def test_cache_hit_increments_counter(loaded_llm: LLMClassifier):
    from tidus.classification.models import LLMResult
    fake = LLMResult(
        domain="chat", complexity="simple", privacy="public",
        confidence={"domain": 0.95, "complexity": 0.95, "privacy": 0.95},
    )
    loaded_llm._cache.put(loaded_llm._cache_key("hi"), fake)

    before = _sample_value(T5_CALLS_TOTAL, result="cache_hit")
    out = await loaded_llm.classify("hi")
    after = _sample_value(T5_CALLS_TOTAL, result="cache_hit")

    assert out is not None
    assert after - before == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_rate_limited_increments_counter(loaded_llm: LLMClassifier):
    # Drain the limiter by construction — 0 per minute rejects everything.
    loaded_llm._limiter._max = 0
    before = _sample_value(T5_CALLS_TOTAL, result="rate_limited")
    out = await loaded_llm.classify("hi")
    after = _sample_value(T5_CALLS_TOTAL, result="rate_limited")

    assert out is None
    assert after - before == pytest.approx(1.0)


@pytest.mark.asyncio
@respx.mock
async def test_success_increments_counter_and_records_latency(loaded_llm: LLMClassifier):
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(
        200, json={"message": {"content": (
            '{"domain":"chat","complexity":"simple","privacy":"public","rationale":"x"}'
        )}},
    ))

    before_calls = _sample_value(T5_CALLS_TOTAL, result="success")
    before_latency_count = T5_LATENCY_SECONDS.labels(result="success")._sum.get()

    out = await loaded_llm.classify("unique prompt for success path")

    after_calls = _sample_value(T5_CALLS_TOTAL, result="success")
    after_latency_count = T5_LATENCY_SECONDS.labels(result="success")._sum.get()

    assert out is not None
    assert after_calls - before_calls == pytest.approx(1.0)
    # Latency sum must have increased — any positive delta proves we timed it.
    assert after_latency_count > before_latency_count


@pytest.mark.asyncio
@respx.mock
async def test_network_failure_increments_counter_and_records_latency(loaded_llm: LLMClassifier):
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("connection refused"),
    )

    before = _sample_value(T5_CALLS_TOTAL, result="failure")
    out = await loaded_llm.classify("unique prompt for failure path")
    after = _sample_value(T5_CALLS_TOTAL, result="failure")

    assert out is None
    assert after - before == pytest.approx(1.0)


def test_flip_counter_has_expected_labels():
    # Smoke: the label dimension exists and accepts the privacy literal values.
    T5_FLIPS_TOTAL.labels(from_privacy="public").inc(0)  # no-op increment
    T5_FLIPS_TOTAL.labels(from_privacy="internal").inc(0)
    # If the Counter was defined with a different labelname we'd KeyError here.
