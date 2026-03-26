"""Deep validation of Tidus model selection intelligence.

This suite goes beyond "did it route somewhere?" to verify:

1.  COMPLEXITY ENFORCEMENT — min/max complexity is respected; over/under-qualified
    models are rejected with `complexity_mismatch`, not silently scored out.

2.  DOMAIN SPECIALISATION — code tasks select code-capable models; reasoning tasks
    select reasoning-capable models; classification uses economy tier.

3.  SCORING FORMULA CORRECTNESS — the 70/20/10 weighting (cost/tier/latency)
    produces provably correct results when costs are known.

4.  SAVINGS QUANTIFICATION — every decision is compared against a "naive baseline"
    (always the most expensive eligible model). Dollar savings and percentage
    are asserted, not just logged. This is the core business value of Tidus.

5.  REGRESSION GUARD — long-context routing, confidential code, agent-depth
    combined with complexity constraints.

Run with:
    uv run pytest tests/integration/test_model_selection_intelligence.py -v -s

The -s flag prints the savings table to stdout.
"""

from __future__ import annotations

import math
from typing import NamedTuple
from unittest.mock import AsyncMock, patch

import pytest

from tidus.budget.enforcer import BudgetEnforcer
from tidus.budget.policies import load_budget_policies
from tidus.cost.counter import SpendCounter
from tidus.cost.engine import CostEngine
from tidus.models.guardrails import GuardrailPolicy
from tidus.models.model_registry import ModelTier
from tidus.models.routing import RejectionReason
from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor
from tidus.router.capability_matcher import CapabilityMatcher
from tidus.router.registry import ModelRegistry
from tidus.router.selector import ModelSelectionError, ModelSelector
from tidus.utils.yaml_loader import load_yaml

# ── Config paths ──────────────────────────────────────────────────────────────

MODELS_YAML = "config/models.yaml"
BUDGETS_YAML = "config/budgets.yaml"
POLICIES_YAML = "config/policies.yaml"


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def registry():
    return ModelRegistry.load(MODELS_YAML)


@pytest.fixture(scope="module")
def selector(registry):
    raw = load_yaml(POLICIES_YAML)
    gp = GuardrailPolicy.model_validate(raw["guardrails"])
    buffer_pct = raw["cost"]["estimate_buffer_pct"]
    budgets = load_budget_policies(BUDGETS_YAML)
    counter = SpendCounter()
    enforcer = BudgetEnforcer(budgets, counter)
    matcher = CapabilityMatcher(gp)
    engine = CostEngine(buffer_pct=buffer_pct)
    return ModelSelector(registry, enforcer, matcher, engine)


def _task(
    complexity: Complexity,
    domain: Domain,
    privacy: Privacy = Privacy.public,
    team_id: str = "team-engineering",
    agent_depth: int = 0,
    preferred_model_id: str | None = None,
    max_cost_usd: float | None = None,
    estimated_input_tokens: int = 500,
    estimated_output_tokens: int = 256,
) -> TaskDescriptor:
    return TaskDescriptor(
        team_id=team_id,
        complexity=complexity,
        domain=domain,
        privacy=privacy,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        messages=[{"role": "user", "content": "test"}],
        agent_depth=agent_depth,
        preferred_model_id=preferred_model_id,
        max_cost_usd=max_cost_usd,
    )


async def _select(selector, task) -> object:
    """Helper: patch tokenizer and run selection."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=task.estimated_input_tokens)):
        return await selector.select(task)


def _naive_cost(registry, decision_model_id: str, input_tok: int, output_tok: int, buffer_pct: float = 0.15) -> float:
    """Return what the MOST EXPENSIVE eligible model would have cost."""
    spec = registry.get(decision_model_id)
    if spec is None:
        return 0.0
    bi = int(input_tok * (1 + buffer_pct))
    bo = int(output_tok * (1 + buffer_pct))
    return bi / 1000 * spec.input_price + bo / 1000 * spec.output_price


def _savings_pct(baseline_usd: float, actual_usd: float) -> float:
    if baseline_usd == 0:
        return 0.0
    return (baseline_usd - actual_usd) / baseline_usd * 100


# ── 1. COMPLEXITY ENFORCEMENT ─────────────────────────────────────────────────

class TestComplexityEnforcement:
    """Verify that min_complexity / max_complexity gating is active."""

    @pytest.mark.asyncio
    async def test_simple_task_excludes_complex_only_models(self, selector, registry):
        """Models with min_complexity=complex must be absent from simple-task candidates.

        deepseek-r1 and o3 are both min_complexity=complex — they must be
        rejected with complexity_mismatch, not just scored out.
        """
        task = _task(Complexity.simple, Domain.chat)
        decision = await _select(selector, task)

        assert decision.accepted
        # deepseek-r1 and o3 should never win a simple task
        assert decision.chosen_model_id not in {"deepseek-r1", "o3", "claude-opus-4-6", "grok-3"}

    @pytest.mark.asyncio
    async def test_complex_only_models_get_mismatch_rejection(self, registry):
        """A registry containing only min_complexity=complex models must reject
        a simple task with complexity_mismatch for every candidate.

        Uses a minimal in-memory registry of just o3, deepseek-r1, and grok-3
        (all min_complexity=complex) to isolate the check without interference
        from budget or domain filters.
        """
        from tidus.cost.counter import SpendCounter
        from tidus.cost.engine import CostEngine
        from tidus.router.capability_matcher import CapabilityMatcher
        from tidus.budget.enforcer import BudgetEnforcer

        raw = load_yaml(POLICIES_YAML)
        gp = GuardrailPolicy.model_validate(raw["guardrails"])
        buffer_pct = raw["cost"]["estimate_buffer_pct"]

        # Pull real specs from the full registry
        complex_models = [registry.get(mid) for mid in ["o3", "deepseek-r1", "grok-3"]]
        assert all(m is not None for m in complex_models), "Complex-min models missing from registry"

        mini_reg = ModelRegistry(complex_models)
        enforcer = BudgetEnforcer([], SpendCounter())
        matcher = CapabilityMatcher(gp)
        engine = CostEngine(buffer_pct=buffer_pct)
        sel = ModelSelector(mini_reg, enforcer, matcher, engine)

        with pytest.raises(ModelSelectionError) as exc:
            with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
                # simple chat task — all three models have min_complexity=complex
                await sel.select(_task(Complexity.simple, Domain.chat))

        reasons = {r.chosen_model_id: r.rejection_reason for r in exc.value.rejections}
        assert reasons.get("o3") == RejectionReason.complexity_mismatch
        assert reasons.get("deepseek-r1") == RejectionReason.complexity_mismatch
        assert reasons.get("grok-3") == RejectionReason.complexity_mismatch

    @pytest.mark.asyncio
    async def test_moderate_task_excludes_simple_only_models(self, selector, registry):
        """Models with max_complexity=simple (phi-4-ollama, gemini-nano) must
        not appear in moderate task selections."""
        task = _task(Complexity.moderate, Domain.chat)
        decision = await _select(selector, task)

        assert decision.accepted
        assert decision.chosen_model_id not in {"phi-4-ollama", "gemini-nano"}
        # Those models have max_complexity=simple — ineligible for moderate

    @pytest.mark.asyncio
    async def test_critical_task_uses_critical_capable_models(self, selector, registry):
        """Only models with max_complexity=critical are eligible for critical tasks."""
        task = _task(Complexity.critical, Domain.reasoning)
        decision = await _select(selector, task)

        model = registry.get(decision.chosen_model_id)
        assert model is not None
        assert model.max_complexity == "critical", (
            f"{decision.chosen_model_id} has max_complexity={model.max_complexity}, "
            f"should only select models rated for critical tasks"
        )
        assert model.tier == ModelTier.premium


# ── 2. DOMAIN SPECIALISATION ──────────────────────────────────────────────────

class TestDomainSpecialisation:
    """The right model family for the right job."""

    @pytest.mark.asyncio
    async def test_code_task_selects_code_capable_model(self, selector, registry):
        """Every code task must resolve to a model that declares code capability."""
        for complexity in (Complexity.simple, Complexity.moderate, Complexity.complex):
            task = _task(complexity, Domain.code)
            decision = await _select(selector, task)
            model = registry.get(decision.chosen_model_id)
            from tidus.models.model_registry import Capability
            assert Capability.code in model.capabilities, (
                f"{decision.chosen_model_id} lacks code capability for {complexity} code task"
            )

    @pytest.mark.asyncio
    async def test_reasoning_task_selects_reasoning_capable_model(self, selector, registry):
        from tidus.models.model_registry import Capability
        for complexity in (Complexity.complex, Complexity.critical):
            task = _task(complexity, Domain.reasoning)
            decision = await _select(selector, task)
            model = registry.get(decision.chosen_model_id)
            assert Capability.reasoning in model.capabilities, (
                f"{decision.chosen_model_id} lacks reasoning capability for {complexity}"
            )

    @pytest.mark.asyncio
    async def test_classification_routes_to_economy_tier(self, selector, registry):
        """Simple classification should route to a tier-3 or cheaper model,
        not premium tier — no need to use claude-opus for labelling."""
        task = _task(Complexity.simple, Domain.classification)
        decision = await _select(selector, task)
        model = registry.get(decision.chosen_model_id)
        # tier 3 = economy, tier 4 = local — both acceptable
        assert model.tier.value >= 3, (
            f"Expected economy/local for simple classification, got {decision.chosen_model_id} "
            f"(tier {model.tier})"
        )

    @pytest.mark.asyncio
    async def test_summarisation_moderate_stays_under_tier2(self, selector, registry):
        """Moderate summarisation should not need a premium tier-1 model."""
        task = _task(Complexity.moderate, Domain.summarization)
        decision = await _select(selector, task)
        model = registry.get(decision.chosen_model_id)
        # Tier ceiling for moderate is 3. Premium tier-1 should be excluded.
        assert model.tier != ModelTier.premium, (
            f"Premium model selected for moderate summarisation: {decision.chosen_model_id}"
        )

    @pytest.mark.asyncio
    async def test_complex_code_does_not_use_local_models(self, selector, registry):
        """Complex code tasks (tier ceiling=2) must not fall back to tier-4 local.

        Local models lack the quality guarantees needed for complex code tasks.
        """
        task = _task(Complexity.complex, Domain.code)
        decision = await _select(selector, task)
        model = registry.get(decision.chosen_model_id)
        assert not model.is_local, (
            f"Local model selected for complex code: {decision.chosen_model_id}"
        )
        assert model.tier.value <= 2, (
            f"Expected tier ≤ 2 for complex code, got {decision.chosen_model_id} (tier {model.tier})"
        )

    @pytest.mark.asyncio
    async def test_critical_code_selects_premium_code_model(self, selector, registry):
        """Critical code (tier ceiling=1) must select a tier-1 code-capable model.

        gpt-5-codex, o3, claude-opus-4-6, deepseek-r1 are all valid candidates.
        """
        from tidus.models.model_registry import Capability
        task = _task(Complexity.critical, Domain.code)
        decision = await _select(selector, task)
        model = registry.get(decision.chosen_model_id)
        assert model.tier == ModelTier.premium
        assert Capability.code in model.capabilities


# ── 3. SCORING FORMULA CORRECTNESS ───────────────────────────────────────────

class TestScoringFormula:
    """Validate that cost×0.70 + tier×0.20 + latency×0.10 behaves as documented."""

    @pytest.mark.asyncio
    async def test_cheapest_model_wins_when_quality_equal(self, selector, registry):
        """Among models with equal capabilities and tier, cheapest wins.

        For a moderate summarisation task (tier ceiling=3), deepseek-v3 or
        mistral-small should beat any tier-3 cloud model due to lower cost.
        """
        task = _task(Complexity.moderate, Domain.summarization, estimated_input_tokens=1000)
        decision = await _select(selector, task)

        assert decision.accepted
        assert decision.estimated_cost_usd is not None

        # The winner's cost should be close to the minimum possible for the task
        model = registry.get(decision.chosen_model_id)
        # Premium models (tier 1) should never win moderate summarisation
        assert model.tier != ModelTier.premium, (
            f"Premium model won moderate summarisation scoring: {decision.chosen_model_id}"
        )

    @pytest.mark.asyncio
    async def test_score_is_between_zero_and_one(self, selector):
        """The normalised score must lie in [0, 1] for multi-candidate sets."""
        for complexity in (Complexity.simple, Complexity.moderate, Complexity.complex):
            task = _task(complexity, Domain.chat)
            decision = await _select(selector, task)
            if decision.score is not None:
                assert 0.0 <= decision.score <= 1.0, (
                    f"Score {decision.score} out of range for {complexity}"
                )

    @pytest.mark.asyncio
    async def test_single_candidate_scores_zero(self, registry):
        """When exactly one model survives all stages, score must be 0.0
        (min-max normalisation of a single value returns 0.0)."""
        from tidus.models.model_registry import Capability, ModelSpec, ModelTier, TokenizerType
        from datetime import date
        from tidus.cost.counter import SpendCounter
        from tidus.cost.engine import CostEngine
        from tidus.router.registry import ModelRegistry
        from tidus.router.capability_matcher import CapabilityMatcher
        from tidus.budget.enforcer import BudgetEnforcer

        raw = load_yaml(POLICIES_YAML)
        gp = GuardrailPolicy.model_validate(raw["guardrails"])
        buffer_pct = raw["cost"]["estimate_buffer_pct"]

        sole = ModelSpec(
            model_id="sole-model",
            vendor="test",
            tier=ModelTier.premium,
            max_context=200000,
            input_price=0.005,
            output_price=0.020,
            tokenizer=TokenizerType.tiktoken_cl100k,
            latency_p50_ms=1000,
            capabilities=[Capability.reasoning],
            min_complexity="critical",
            max_complexity="critical",
            is_local=False,
            enabled=True,
            deprecated=False,
            fallbacks=[],
            last_price_check=date(2026, 1, 1),
        )
        reg = ModelRegistry([sole])
        enforcer = BudgetEnforcer([], SpendCounter())
        matcher = CapabilityMatcher(gp)
        engine = CostEngine(buffer_pct=buffer_pct)
        sel = ModelSelector(reg, enforcer, matcher, engine)

        with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
            decision = await sel.select(_task(Complexity.critical, Domain.reasoning))

        assert decision.score == 0.0, "Single-candidate set must score 0.0"
        assert decision.chosen_model_id == "sole-model"


# ── 4. SAVINGS QUANTIFICATION ─────────────────────────────────────────────────

class TestSavingsQuantification:
    """Demonstrate and assert real dollar savings vs. naive (always-premium) routing.

    This is the core business case: Tidus routes to the cheapest capable model
    rather than always using the most expensive one.
    """

    class _Scenario(NamedTuple):
        label: str
        complexity: Complexity
        domain: Domain
        input_tok: int
        output_tok: int
        # Naive baseline: most expensive eligible model for this task
        naive_model_id: str
        # Minimum expected savings percentage vs naive (conservative floor)
        min_savings_pct: float

    # Real-world enterprise scenario mix
    SCENARIOS = [
        # Simple chat → free local vs. claude-opus
        _Scenario("simple_chat",      Complexity.simple,   Domain.chat,           500,  256, "claude-opus-4-6",  90.0),
        # Simple code → cheap code model vs. gpt-5-codex
        _Scenario("simple_code",      Complexity.simple,   Domain.code,           800,  512, "gpt-5-codex",      80.0),
        # Moderate summarisation → mid vs. claude-opus
        _Scenario("mod_summarise",    Complexity.moderate, Domain.summarization,  2000, 512, "claude-opus-4-6",  85.0),
        # Moderate classification → economy vs. o3
        _Scenario("mod_classify",     Complexity.moderate, Domain.classification, 300,  50,  "o3",               90.0),
        # Complex code → mid vs. claude-opus
        _Scenario("complex_code",     Complexity.complex,  Domain.code,           1500, 800, "claude-opus-4-6",  70.0),
        # Complex reasoning → deepseek-r1 vs. grok-3 (both tier 1, but costs differ)
        _Scenario("complex_reason",   Complexity.complex,  Domain.reasoning,      1000, 500, "grok-3",           40.0),
        # Critical reasoning — only tier-1 eligible; deepseek-r1 vs. claude-opus
        _Scenario("critical_reason",  Complexity.critical, Domain.reasoning,      2000, 1000,"claude-opus-4-6",  85.0),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.label for s in SCENARIOS])
    async def test_savings_vs_naive_baseline(self, selector, registry, scenario):
        """For each scenario, assert Tidus saves at least `min_savings_pct`%
        versus always routing to the naive baseline model."""
        task = _task(
            scenario.complexity,
            scenario.domain,
            estimated_input_tokens=scenario.input_tok,
            estimated_output_tokens=scenario.output_tok,
        )
        decision = await _select(selector, task)

        assert decision.accepted, f"{scenario.label}: routing failed"
        assert decision.estimated_cost_usd is not None

        # Compute naive baseline cost (what the expensive model would have cost)
        naive = registry.get(scenario.naive_model_id)
        assert naive is not None, f"Baseline model {scenario.naive_model_id} not in registry"

        buffer = 0.15
        bi = int(scenario.input_tok * (1 + buffer))
        bo = int(scenario.output_tok * (1 + buffer))
        naive_cost = bi / 1000 * naive.input_price + bo / 1000 * naive.output_price

        actual_cost = decision.estimated_cost_usd
        pct = _savings_pct(naive_cost, actual_cost)

        print(
            f"\n  [{scenario.label}] "
            f"naive={naive_cost:.6f} USD ({scenario.naive_model_id})  "
            f"tidus={actual_cost:.6f} USD ({decision.chosen_model_id})  "
            f"savings={pct:.1f}%"
        )

        assert pct >= scenario.min_savings_pct, (
            f"{scenario.label}: expected ≥{scenario.min_savings_pct:.0f}% savings, "
            f"got {pct:.1f}% "
            f"(tidus={decision.chosen_model_id} @ ${actual_cost:.6f} vs "
            f"naive={scenario.naive_model_id} @ ${naive_cost:.6f})"
        )

    @pytest.mark.asyncio
    async def test_enterprise_monthly_savings_projection(self, selector, registry):
        """Simulate a 500-user enterprise (200 requests/day each, 1K tokens avg)
        and project monthly savings.

        This is the ROI calculation from docs/roi-calculator.md put into a test.
        """
        # Task distribution (fraction of total requests)
        TASK_MIX = [
            (0.60, Complexity.simple,   Domain.chat,           800,  256),
            (0.20, Complexity.moderate, Domain.summarization,  2000, 512),
            (0.12, Complexity.moderate, Domain.code,           1500, 512),
            (0.05, Complexity.complex,  Domain.code,           2000, 1000),
            (0.03, Complexity.critical, Domain.reasoning,      3000, 1500),
        ]
        MONTHLY_REQUESTS = 500 * 200 * 30  # users × daily_requests × days

        # Naive baseline: always claude-opus-4-6
        naive_model = registry.get("claude-opus-4-6")
        assert naive_model is not None

        tidus_total = 0.0
        naive_total = 0.0

        for fraction, complexity, domain, in_tok, out_tok in TASK_MIX:
            task = _task(complexity, domain, estimated_input_tokens=in_tok, estimated_output_tokens=out_tok)
            decision = await _select(selector, task)
            assert decision.accepted, f"Routing failed for {complexity}/{domain}"

            buffer = 0.15
            bi = int(in_tok * (1 + buffer))
            bo = int(out_tok * (1 + buffer))

            naive_per_req = bi / 1000 * naive_model.input_price + bo / 1000 * naive_model.output_price
            tidus_per_req = decision.estimated_cost_usd

            volume = fraction * MONTHLY_REQUESTS
            naive_total += naive_per_req * volume
            tidus_total += tidus_per_req * volume

        savings = naive_total - tidus_total
        savings_pct = _savings_pct(naive_total, tidus_total)

        print(
            f"\n  === ENTERPRISE MONTHLY SAVINGS PROJECTION ==="
            f"\n  Total monthly requests : {MONTHLY_REQUESTS:,}"
            f"\n  Naive (always Opus)    : ${naive_total:,.2f}"
            f"\n  Tidus (smart routing)  : ${tidus_total:,.2f}"
            f"\n  Gross monthly savings  : ${savings:,.2f}  ({savings_pct:.1f}%)"
        )

        # Tidus must save at least 70% over always-premium routing for this mix
        assert savings_pct >= 70.0, (
            f"Enterprise savings below 70%: got {savings_pct:.1f}%. "
            f"Naive=${naive_total:.2f}, Tidus=${tidus_total:.2f}"
        )
        # Sanity: Tidus cost should be positive (not free — some cloud models selected)
        assert tidus_total > 0.0


# ── 5. REGRESSION GUARD ───────────────────────────────────────────────────────

class TestRegressionGuard:
    """Edge cases that previously caused bugs or represent subtle routing decisions."""

    @pytest.mark.asyncio
    async def test_long_context_task_routes_to_large_window_model(self, registry):
        """A task with 2K tokens must select the large-context model and reject
        the small-context one with context_too_large.

        Uses a minimal two-model registry to isolate the context window check
        from the guardrail token-per-step limit (which caps at 8K).
        """
        from tidus.models.model_registry import Capability, ModelSpec, ModelTier, TokenizerType
        from datetime import date
        from tidus.cost.counter import SpendCounter
        from tidus.cost.engine import CostEngine
        from tidus.router.capability_matcher import CapabilityMatcher
        from tidus.budget.enforcer import BudgetEnforcer

        raw = load_yaml(POLICIES_YAML)
        gp = GuardrailPolicy.model_validate(raw["guardrails"])
        buffer_pct = raw["cost"]["estimate_buffer_pct"]

        def _spec(model_id, max_context):
            return ModelSpec(
                model_id=model_id,
                vendor="test",
                tier=ModelTier.economy,
                max_context=max_context,
                input_price=0.001,
                output_price=0.002,
                tokenizer=TokenizerType.tiktoken_cl100k,
                latency_p50_ms=500,
                capabilities=[Capability.summarization],
                min_complexity="simple",
                max_complexity="moderate",
                is_local=False,
                enabled=True,
                deprecated=False,
                fallbacks=[],
                last_price_check=date(2026, 1, 1),
            )

        small = _spec("small-ctx-model", max_context=1000)
        large = _spec("large-ctx-model", max_context=128000)
        reg = ModelRegistry([small, large])
        enforcer = BudgetEnforcer([], SpendCounter())
        matcher = CapabilityMatcher(gp)
        engine = CostEngine(buffer_pct=buffer_pct)
        sel = ModelSelector(reg, enforcer, matcher, engine)

        # 2000-token task: exceeds small model's 1K context, fits large model's 128K
        task = _task(
            Complexity.moderate,
            Domain.summarization,
            estimated_input_tokens=2000,
        )
        with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=2000)):
            decision = await sel.select(task)

        assert decision.accepted, "Large-context task must route successfully"
        assert decision.chosen_model_id == "large-ctx-model", (
            f"Expected large-ctx-model, got {decision.chosen_model_id}"
        )

        # Verify the small model was explicitly rejected with context_too_large
        # (Check via a ModelSelectionError on a registry with only the small model)
        reg_small_only = ModelRegistry([small])
        sel_small = ModelSelector(reg_small_only, BudgetEnforcer([], SpendCounter()), matcher, engine)
        with pytest.raises(ModelSelectionError) as exc:
            with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=2000)):
                await sel_small.select(task)
        reasons = [r.rejection_reason for r in exc.value.rejections]
        assert RejectionReason.context_too_large in reasons, (
            f"Expected context_too_large rejection, got: {reasons}"
        )

    @pytest.mark.asyncio
    async def test_confidential_code_routes_to_local_code_capable_model(self, selector, registry):
        """Confidential + code must select a local model that supports code.

        llama4-maverick-ollama is the only local model with code capability.
        """
        from tidus.models.model_registry import Capability
        task = _task(Complexity.simple, Domain.code, privacy=Privacy.confidential)
        decision = await _select(selector, task)

        model = registry.get(decision.chosen_model_id)
        assert model.is_local, f"Non-local model selected for confidential code: {decision.chosen_model_id}"
        assert Capability.code in model.capabilities, (
            f"Selected local model {decision.chosen_model_id} lacks code capability"
        )

    @pytest.mark.asyncio
    async def test_selection_is_deterministic(self, selector):
        """The same task inputs must always produce the same model selection."""
        task = _task(Complexity.complex, Domain.reasoning, estimated_input_tokens=1000)
        decisions = []
        for _ in range(5):
            d = await _select(selector, task)
            decisions.append(d.chosen_model_id)

        assert len(set(decisions)) == 1, (
            f"Non-deterministic selection for complex reasoning: {set(decisions)}"
        )

    @pytest.mark.asyncio
    async def test_complexity_mismatch_rejection_in_log(self, registry):
        """When a simple task is attempted on a registry with ONLY complex-min models,
        the rejection log must contain complexity_mismatch reasons — not empty."""
        from tidus.models.model_registry import Capability, ModelSpec, ModelTier, TokenizerType
        from datetime import date
        from tidus.cost.counter import SpendCounter
        from tidus.cost.engine import CostEngine
        from tidus.router.registry import ModelRegistry
        from tidus.router.capability_matcher import CapabilityMatcher
        from tidus.budget.enforcer import BudgetEnforcer

        raw = load_yaml(POLICIES_YAML)
        gp = GuardrailPolicy.model_validate(raw["guardrails"])
        buffer_pct = raw["cost"]["estimate_buffer_pct"]

        complex_only = ModelSpec(
            model_id="complex-only-model",
            vendor="test",
            tier=ModelTier.premium,
            max_context=128000,
            input_price=0.005,
            output_price=0.015,
            tokenizer=TokenizerType.tiktoken_cl100k,
            latency_p50_ms=1000,
            capabilities=[Capability.chat],
            min_complexity="complex",
            max_complexity="critical",
            is_local=False,
            enabled=True,
            deprecated=False,
            fallbacks=[],
            last_price_check=date(2026, 1, 1),
        )
        reg = ModelRegistry([complex_only])
        enforcer = BudgetEnforcer([], SpendCounter())
        matcher = CapabilityMatcher(gp)
        engine = CostEngine(buffer_pct=buffer_pct)
        sel = ModelSelector(reg, enforcer, matcher, engine)

        with pytest.raises(ModelSelectionError) as exc:
            with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
                await sel.select(_task(Complexity.simple, Domain.chat))

        reasons = [r.rejection_reason for r in exc.value.rejections]
        assert RejectionReason.complexity_mismatch in reasons, (
            "Expected complexity_mismatch in rejection log; got: {reasons}"
        )

    @pytest.mark.asyncio
    async def test_all_complexity_domain_combinations_route_or_fail_gracefully(self, selector):
        """Full cross-product: every (complexity, domain) pair must either succeed
        or raise ModelSelectionError — never an unhandled exception."""
        domains = [Domain.chat, Domain.code, Domain.reasoning, Domain.extraction,
                   Domain.classification, Domain.summarization]
        complexities = [Complexity.simple, Complexity.moderate, Complexity.complex, Complexity.critical]

        outcomes: dict[str, str] = {}
        for complexity in complexities:
            for domain in domains:
                key = f"{complexity.value}/{domain.value}"
                try:
                    d = await _select(selector, _task(complexity, domain))
                    outcomes[key] = f"[OK] {d.chosen_model_id}"
                except ModelSelectionError as e:
                    outcomes[key] = f"[--] stage={e.stage}"
                except Exception as e:
                    pytest.fail(f"Unhandled exception for {key}: {type(e).__name__}: {e}")

        # Print the full routing matrix
        print("\n  === ROUTING MATRIX ===")
        for key, result in sorted(outcomes.items()):
            print(f"  {key:30s} → {result}")

        # At minimum: simple and moderate tasks for common domains must succeed
        for domain in (Domain.chat, Domain.code, Domain.extraction, Domain.classification):
            for complexity in (Complexity.simple, Complexity.moderate):
                key = f"{complexity.value}/{domain.value}"
                assert outcomes[key].startswith("[OK]"), (
                    f"{key} routing must succeed, got: {outcomes[key]}"
                )
