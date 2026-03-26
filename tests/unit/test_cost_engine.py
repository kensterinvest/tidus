"""Unit tests for CostEngine — safety buffer math and per-vendor dispatch."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from tidus.cost.engine import CostEngine
from tidus.models.model_registry import Capability, ModelSpec, ModelTier, TokenizerType
from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor


def _spec(model_id: str, input_price: float, output_price: float,
          tokenizer: TokenizerType = TokenizerType.tiktoken_cl100k) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        vendor="test",
        tier=ModelTier(3),
        max_context=128000,
        input_price=input_price,
        output_price=output_price,
        tokenizer=tokenizer,
        latency_p50_ms=400,
        capabilities=[Capability.chat],
        min_complexity="simple",
        max_complexity="critical",
        last_price_check=date(2025, 1, 1),
    )


def _task(input_tokens: int = 1000, output_tokens: int = 500) -> TaskDescriptor:
    return TaskDescriptor(
        team_id="team-test",
        complexity=Complexity.simple,
        domain=Domain.chat,
        privacy=Privacy.public,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        messages=[{"role": "user", "content": "x" * 100}],
    )


@pytest.mark.asyncio
async def test_buffer_applied_to_cost():
    """Cost should include the 15% safety buffer on both input and output."""
    engine = CostEngine(buffer_pct=0.15)
    model = _spec("test-model", input_price=0.001, output_price=0.002)

    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=1000)):
        estimate = await engine.estimate(model, _task(input_tokens=1000, output_tokens=500))

    buffered_input = int(1000 * 1.15)   # 1150
    buffered_output = int(500 * 1.15)   # 575
    expected_cost = (buffered_input / 1000 * 0.001) + (buffered_output / 1000 * 0.002)

    assert estimate.buffered_input_tokens == buffered_input
    assert estimate.buffered_output_tokens == buffered_output
    assert abs(estimate.estimated_cost_usd - expected_cost) < 1e-9
    assert estimate.buffer_pct == 0.15


@pytest.mark.asyncio
async def test_zero_buffer():
    """buffer_pct=0.0 means raw counts drive the cost directly."""
    engine = CostEngine(buffer_pct=0.0)
    model = _spec("free-model", input_price=0.0, output_price=0.0)

    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=500)):
        estimate = await engine.estimate(model, _task(input_tokens=500, output_tokens=200))

    assert estimate.estimated_cost_usd == 0.0
    assert estimate.buffered_input_tokens == 500
    assert estimate.buffered_output_tokens == 200


@pytest.mark.asyncio
async def test_local_model_zero_cost():
    """Local (Ollama) models have 0.0 pricing — cost should be exactly 0."""
    engine = CostEngine(buffer_pct=0.15)
    model = _spec("llama4-scout", input_price=0.0, output_price=0.0,
                  tokenizer=TokenizerType.ollama)

    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=800)):
        estimate = await engine.estimate(model, _task())

    assert estimate.estimated_cost_usd == 0.0


def test_invalid_buffer_raises():
    with pytest.raises(ValueError, match="buffer_pct"):
        CostEngine(buffer_pct=1.5)


def test_estimate_from_counts_sync():
    """estimate_from_counts should work without async and apply buffer correctly."""
    engine = CostEngine(buffer_pct=0.10)
    model = _spec("sync-model", input_price=0.002, output_price=0.008)
    estimate = engine.estimate_from_counts(model, input_tokens=2000, output_tokens=300)

    buffered_input = int(2000 * 1.10)
    buffered_output = int(300 * 1.10)
    expected = (buffered_input / 1000 * 0.002) + (buffered_output / 1000 * 0.008)
    assert abs(estimate.estimated_cost_usd - expected) < 1e-9
