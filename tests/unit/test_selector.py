"""Unit tests for the 5-stage model selector.

All tests use a small in-memory registry and mocked cost engine so they
run fast with no network calls or vendor SDK dependencies.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from tidus.budget.enforcer import BudgetEnforcer
from tidus.cost.counter import SpendCounter
from tidus.cost.engine import CostEngine
from tidus.models.budget import BudgetPeriod, BudgetPolicy, BudgetScope
from tidus.models.cost import CostEstimate
from tidus.models.guardrails import GuardrailPolicy
from tidus.models.model_registry import Capability, ModelSpec, ModelTier, TokenizerType
from tidus.models.routing import RejectionReason
from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor
from tidus.router.capability_matcher import CapabilityMatcher
from tidus.router.registry import ModelRegistry
from tidus.router.selector import ModelSelectionError, ModelSelector

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_spec(
    model_id: str,
    tier: int,
    input_price: float,
    output_price: float,
    capabilities: list[Capability] | None = None,
    is_local: bool = False,
    enabled: bool = True,
    deprecated: bool = False,
    min_complexity: str = "simple",
    max_complexity: str = "critical",
    latency_p50_ms: int = 500,
    max_context: int = 128000,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        vendor="test",
        tier=ModelTier(tier),
        max_context=max_context,
        input_price=input_price,
        output_price=output_price,
        tokenizer=TokenizerType.tiktoken_cl100k,
        latency_p50_ms=latency_p50_ms,
        capabilities=capabilities or [Capability.chat],
        min_complexity=min_complexity,
        max_complexity=max_complexity,
        is_local=is_local,
        enabled=enabled,
        deprecated=deprecated,
        fallbacks=[],
        last_price_check=date(2025, 1, 1),
    )


def _make_task(
    complexity: Complexity = Complexity.simple,
    domain: Domain = Domain.chat,
    privacy: Privacy = Privacy.public,
    estimated_input_tokens: int = 200,
    estimated_output_tokens: int = 256,
    team_id: str = "team-eng",
    max_cost_usd: float | None = None,
    preferred_model_id: str | None = None,
    agent_depth: int = 0,
) -> TaskDescriptor:
    return TaskDescriptor(
        team_id=team_id,
        complexity=complexity,
        domain=domain,
        privacy=privacy,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        messages=[{"role": "user", "content": "hello"}],
        max_cost_usd=max_cost_usd,
        preferred_model_id=preferred_model_id,
        agent_depth=agent_depth,
    )


def _mock_engine(cost_per_call: float = 0.001) -> CostEngine:
    """Return a CostEngine whose estimate() always returns a fixed cost."""
    engine = MagicMock(spec=CostEngine)
    engine.estimate = AsyncMock(
        return_value=CostEstimate(
            model_id="any",
            raw_input_tokens=200,
            raw_output_tokens=256,
            buffered_input_tokens=230,
            buffered_output_tokens=295,
            estimated_cost_usd=cost_per_call,
            buffer_pct=0.15,
        )
    )
    return engine


def _build_selector(
    specs: list[ModelSpec],
    policies: list[BudgetPolicy] | None = None,
    cost_per_call: float = 0.001,
    guardrail_policy: GuardrailPolicy | None = None,
) -> ModelSelector:
    registry = ModelRegistry(specs)
    counter = SpendCounter()
    enforcer = BudgetEnforcer(policies or [], counter)
    gp = guardrail_policy or GuardrailPolicy()
    matcher = CapabilityMatcher(gp)
    engine = _mock_engine(cost_per_call)
    return ModelSelector(registry, enforcer, matcher, engine)


# ── Stage 1: Hard constraints ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disabled_model_is_rejected():
    specs = [_make_spec("disabled-model", tier=3, input_price=0.0001, output_price=0.0003, enabled=False)]
    selector = _build_selector(specs)
    with pytest.raises(ModelSelectionError) as exc_info:
        await selector.select(_make_task())
    assert exc_info.value.stage == 2
    rejections = {r.chosen_model_id: r.rejection_reason for r in exc_info.value.rejections}
    assert rejections["disabled-model"] == RejectionReason.model_disabled


@pytest.mark.asyncio
async def test_context_too_large_rejected():
    specs = [_make_spec("small-ctx", tier=3, input_price=0.0001, output_price=0.0003, max_context=100)]
    selector = _build_selector(specs)
    task = _make_task(estimated_input_tokens=500)
    with pytest.raises(ModelSelectionError):
        await selector.select(task)


@pytest.mark.asyncio
async def test_domain_not_supported_rejected():
    # Model only supports chat; task needs code
    specs = [_make_spec("chat-only", tier=3, input_price=0.0001, output_price=0.0003,
                         capabilities=[Capability.chat])]
    selector = _build_selector(specs)
    task = _make_task(domain=Domain.code)
    with pytest.raises(ModelSelectionError) as exc_info:
        await selector.select(task)
    rejections = {r.chosen_model_id: r.rejection_reason for r in exc_info.value.rejections}
    assert rejections["chat-only"] == RejectionReason.domain_not_supported


@pytest.mark.asyncio
async def test_confidential_task_requires_local_model():
    cloud_spec = _make_spec("cloud-model", tier=2, input_price=0.001, output_price=0.003, is_local=False)
    local_spec = _make_spec("local-model", tier=4, input_price=0.0, output_price=0.0, is_local=True)
    selector = _build_selector([cloud_spec, local_spec])
    task = _make_task(privacy=Privacy.confidential)
    decision = await selector.select(task)
    assert decision.chosen_model_id == "local-model"
    assert decision.accepted


# ── Stage 3: Complexity tier ceiling ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_simple_task_prefers_cheapest_tier():
    """Simple tasks allow all tiers (ceiling=4). With real cost differences, local wins."""
    tier1 = _make_spec("premium", tier=1, input_price=0.015, output_price=0.060)
    tier3 = _make_spec("economy", tier=3, input_price=0.0001, output_price=0.0003)
    tier4 = _make_spec("local", tier=4, input_price=0.0, output_price=0.0, is_local=True)

    # Wire per-model costs so the engine returns accurate prices
    registry = ModelRegistry([tier1, tier3, tier4])
    counter = SpendCounter()
    enforcer = BudgetEnforcer([], counter)
    matcher = CapabilityMatcher(GuardrailPolicy())
    costs = {"premium": 0.015, "economy": 0.0001, "local": 0.0}

    async def side_effect(model, task):
        return CostEstimate(
            model_id=model.model_id,
            raw_input_tokens=200, raw_output_tokens=256,
            buffered_input_tokens=230, buffered_output_tokens=295,
            estimated_cost_usd=costs[model.model_id],
            buffer_pct=0.15,
        )

    engine = MagicMock(spec=CostEngine)
    engine.estimate = side_effect
    selector = ModelSelector(registry, enforcer, matcher, engine)

    task = _make_task(complexity=Complexity.simple)
    decision = await selector.select(task)
    # Premium ($0.015) has a high cost_norm score and should NOT win over cheap options
    assert decision.chosen_model_id != "premium"


@pytest.mark.asyncio
async def test_critical_task_only_uses_tier1():
    tier1 = _make_spec("premium", tier=1, input_price=0.015, output_price=0.060)
    tier2 = _make_spec("mid-tier", tier=2, input_price=0.003, output_price=0.015)
    tier3 = _make_spec("economy", tier=3, input_price=0.0001, output_price=0.0003)
    selector = _build_selector([tier1, tier2, tier3])
    task = _make_task(complexity=Complexity.critical)
    decision = await selector.select(task)
    assert decision.chosen_model_id == "premium"


@pytest.mark.asyncio
async def test_moderate_task_ceiling_excludes_tier4():
    """moderate tasks allow tier ≤ 3 — tier-4 (local/free) models are excluded."""
    tier2 = _make_spec("mid-tier", tier=2, input_price=0.003, output_price=0.015)
    tier3 = _make_spec("economy", tier=3, input_price=0.0001, output_price=0.0003)
    tier4 = _make_spec("local", tier=4, input_price=0.0, output_price=0.0, is_local=True)
    selector = _build_selector([tier2, tier3, tier4])
    task = _make_task(complexity=Complexity.moderate)
    # Tier-4 (local) is excluded by ceiling=3; tier2 and tier3 remain
    decision = await selector.select(task)
    assert decision.chosen_model_id != "local"


# ── Stage 4: Budget enforcement ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_budget_exceeded_raises_selection_error():
    """When estimated cost exceeds team budget, selection should fail."""
    spec = _make_spec("expensive", tier=2, input_price=0.01, output_price=0.04)
    policy = BudgetPolicy(
        policy_id="team-eng-monthly",
        scope=BudgetScope.team,
        scope_id="team-eng",
        period=BudgetPeriod.monthly,
        limit_usd=0.001,  # tiny limit
        hard_stop=True,
    )
    # Pre-fill counter to near limit
    selector = _build_selector([spec], policies=[policy], cost_per_call=0.005)
    task = _make_task()
    with pytest.raises(ModelSelectionError) as exc_info:
        await selector.select(task)
    assert exc_info.value.stage == 4


@pytest.mark.asyncio
async def test_per_request_max_cost_respected():
    """task.max_cost_usd should prevent selection of models that exceed it."""
    cheap = _make_spec("cheap", tier=3, input_price=0.0001, output_price=0.0003)
    # Engine always returns 0.001 cost; set max_cost_usd below that
    selector = _build_selector([cheap], cost_per_call=0.005)
    task = _make_task(max_cost_usd=0.001)
    with pytest.raises(ModelSelectionError) as exc_info:
        await selector.select(task)
    assert exc_info.value.stage == 4


# ── Stage 5: Scoring ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_preferred_model_selected_when_eligible():
    """When preferred_model_id is set and the model passes all stages, use it."""
    spec_a = _make_spec("model-a", tier=2, input_price=0.003, output_price=0.015)
    spec_b = _make_spec("model-b", tier=3, input_price=0.0001, output_price=0.0003)
    selector = _build_selector([spec_a, spec_b])
    task = _make_task(preferred_model_id="model-a")
    decision = await selector.select(task)
    assert decision.chosen_model_id == "model-a"


@pytest.mark.asyncio
async def test_scoring_picks_cheapest_among_equals():
    """When cost is the only differentiator, the cheapest model wins."""
    specs = [
        _make_spec("mid", tier=2, input_price=0.003, output_price=0.015),
        _make_spec("cheap", tier=3, input_price=0.0001, output_price=0.0003),
    ]
    # Use separate cost mocks so cheap is actually cheaper
    registry = ModelRegistry(specs)
    counter = SpendCounter()
    enforcer = BudgetEnforcer([], counter)
    matcher = CapabilityMatcher(GuardrailPolicy())

    engine = MagicMock(spec=CostEngine)

    async def side_effect(model, task):
        costs = {"mid": 0.003, "cheap": 0.0001}
        return CostEstimate(
            model_id=model.model_id,
            raw_input_tokens=200,
            raw_output_tokens=256,
            buffered_input_tokens=230,
            buffered_output_tokens=295,
            estimated_cost_usd=costs[model.model_id],
            buffer_pct=0.15,
        )

    engine.estimate = side_effect
    selector = ModelSelector(registry, enforcer, matcher, engine)
    decision = await selector.select(_make_task(complexity=Complexity.moderate))
    assert decision.chosen_model_id == "cheap"


# ── Guardrails ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_depth_exceeded_raises():
    spec = _make_spec("chat-model", tier=3, input_price=0.0001, output_price=0.0003)
    gp = GuardrailPolicy(max_agent_depth=3)
    selector = _build_selector([spec], guardrail_policy=gp)
    task = _make_task(agent_depth=4)  # exceeds max of 3
    with pytest.raises(ModelSelectionError) as exc_info:
        await selector.select(task)
    rejections = {r.chosen_model_id: r.rejection_reason for r in exc_info.value.rejections}
    assert rejections["chat-model"] == RejectionReason.agent_depth_exceeded


# ── Deprecated model routing (score penalty) ──────────────────────────────────

@pytest.mark.asyncio
async def test_deprecated_model_included_in_routing():
    """Deprecated models must still be routed (only disabled models are excluded).

    If the deprecated model is the only option, it must be returned.
    """
    spec = _make_spec("old-model", tier=2, input_price=0.001, output_price=0.002, deprecated=True)
    selector = _build_selector([spec])
    decision = await selector.select(_make_task())
    assert decision.chosen_model_id == "old-model"


@pytest.mark.asyncio
async def test_deprecated_model_loses_to_equivalent_non_deprecated():
    """A non-deprecated model wins over a deprecated one with identical pricing and tier."""
    deprecated_spec = _make_spec(
        "old-model", tier=2, input_price=0.001, output_price=0.002, deprecated=True
    )
    fresh_spec = _make_spec(
        "new-model", tier=2, input_price=0.001, output_price=0.002, deprecated=False
    )
    selector = _build_selector([deprecated_spec, fresh_spec])
    decision = await selector.select(_make_task())
    assert decision.chosen_model_id == "new-model"


def test_deprecated_model_score_penalty_value():
    """The deprecated score penalty is exactly 0.15 (matches _DEPRECATED_SCORE_PENALTY).

    Tests _score_and_pick() directly — the public selector uses a mock engine that
    assigns the same cost to all models, which would collapse the score spread to 0.
    """
    from tidus.router.selector import _score_and_pick

    deprecated = _make_spec(
        "deprecated", tier=2, input_price=1.0, output_price=2.0, deprecated=True
    )
    fresh = _make_spec(
        "fresh", tier=2, input_price=1.0, output_price=2.0, deprecated=False
    )

    # Single deprecated model: score is always 0.0 (single candidate baseline)
    _, _, score_solo = _score_and_pick([(deprecated, 1.0)])
    assert score_solo == pytest.approx(0.0)

    # Two identical models at same cost/tier/latency → normalised score = 0.0 for both dims;
    # fresh wins with score=0.0; deprecated has effective internal score=0.15 and loses.
    winner_spec, _, winner_score = _score_and_pick([(fresh, 1.0), (deprecated, 1.0)])
    assert winner_spec.model_id == "fresh", "fresh model must beat deprecated at equal cost"
    assert winner_score == pytest.approx(0.0)

    # Same result regardless of input order
    winner_spec2, _, _ = _score_and_pick([(deprecated, 1.0), (fresh, 1.0)])
    assert winner_spec2.model_id == "fresh"


def test_very_cheap_deprecated_beats_expensive_non_deprecated():
    """Deprecated model wins when cost advantage overwhelms the 0.15 penalty.

    Uses _score_and_pick() directly with real cost values so that normalisation
    produces meaningful cost spread — bypasses the mock engine's uniform pricing.
    """
    from tidus.router.selector import _score_and_pick

    cheap_deprecated = _make_spec(
        "cheap-deprecated", tier=3, input_price=0.00001, output_price=0.00002, deprecated=True
    )
    expensive_fresh = _make_spec(
        "expensive-fresh", tier=2, input_price=10.0, output_price=20.0, deprecated=False
    )

    # cheap-deprecated costs $0.00001; expensive-fresh costs $10.00
    winner_spec, _, _ = _score_and_pick([
        (cheap_deprecated, 0.00001),
        (expensive_fresh, 10.0),
    ])
    assert winner_spec.model_id == "cheap-deprecated", (
        "overwhelming cost advantage should outweigh the 0.15 deprecated penalty"
    )
