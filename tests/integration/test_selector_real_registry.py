"""Integration test — ModelSelector against the real config/models.yaml registry.

These tests use the full 26-model registry and real guardrail/budget policies
but mock the tokenizer so no API keys are required.

Run with:
    uv run pytest tests/integration/test_selector_real_registry.py -v

What these tests verify:
  - The YAML configs parse without errors
  - The selector produces sensible routing decisions at each complexity level
  - Privacy.confidential tasks stay on local models
  - Budget hard-stop prevents selection when team spend is exhausted
  - Preferred model override is respected when eligible
  - Agent depth guardrail rejects deep chains
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tidus.budget.enforcer import BudgetEnforcer
from tidus.budget.policies import load_budget_policies
from tidus.cost.counter import SpendCounter
from tidus.cost.engine import CostEngine
from tidus.models.guardrails import GuardrailPolicy
from tidus.models.model_registry import ModelTier
from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor
from tidus.router.capability_matcher import CapabilityMatcher
from tidus.router.registry import ModelRegistry
from tidus.router.selector import ModelSelectionError, ModelSelector
from tidus.utils.yaml_loader import load_yaml

# ── Shared fixtures ───────────────────────────────────────────────────────────

MODELS_YAML = "config/models.yaml"
BUDGETS_YAML = "config/budgets.yaml"
POLICIES_YAML = "config/policies.yaml"


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry.load(MODELS_YAML)


@pytest.fixture(scope="module")
def guardrail_policy():
    raw = load_yaml(POLICIES_YAML)
    return GuardrailPolicy.model_validate(raw["guardrails"])


@pytest.fixture(scope="module")
def buffer_pct():
    raw = load_yaml(POLICIES_YAML)
    return raw["cost"]["estimate_buffer_pct"]


@pytest.fixture
def selector(registry, guardrail_policy, buffer_pct):
    """Fresh selector with a zeroed SpendCounter for each test."""
    budgets = load_budget_policies(BUDGETS_YAML)
    counter = SpendCounter()
    enforcer = BudgetEnforcer(budgets, counter)
    matcher = CapabilityMatcher(guardrail_policy)
    engine = CostEngine(buffer_pct=buffer_pct)
    return ModelSelector(registry, enforcer, matcher, engine)


def _task(
    complexity: Complexity = Complexity.simple,
    domain: Domain = Domain.chat,
    privacy: Privacy = Privacy.public,
    team_id: str = "team-engineering",
    agent_depth: int = 0,
    preferred_model_id: str | None = None,
    max_cost_usd: float | None = None,
    estimated_input_tokens: int = 500,
) -> TaskDescriptor:
    return TaskDescriptor(
        team_id=team_id,
        complexity=complexity,
        domain=domain,
        privacy=privacy,
        estimated_input_tokens=estimated_input_tokens,
        messages=[{"role": "user", "content": "hello"}],
        agent_depth=agent_depth,
        preferred_model_id=preferred_model_id,
        max_cost_usd=max_cost_usd,
    )


# ── Config loading ────────────────────────────────────────────────────────────

def test_registry_loads_all_models(registry):
    """All models defined in models.yaml should load without validation errors.

    43 models are enabled (active adapters); 10 are intentionally disabled
    (pending adapter implementations: Cohere, Groq, Qwen, Perplexity, Together AI).
    """
    assert len(registry) == 53
    enabled = [s for s in registry.list_all() if s.enabled]
    disabled = [s for s in registry.list_all() if not s.enabled]
    assert len(enabled) == 43, f"Expected 43 enabled models, got {len(enabled)}"
    assert len(disabled) == 10, f"Expected 10 disabled (pending) models, got {len(disabled)}"


def test_guardrail_policy_matches_yaml_values(guardrail_policy):
    assert guardrail_policy.max_agent_depth == 5
    assert guardrail_policy.max_tokens_per_step == 8000
    assert guardrail_policy.max_retries_per_task == 3


def test_budget_policies_loaded(buffer_pct):
    budgets = load_budget_policies(BUDGETS_YAML)
    assert len(budgets) >= 3
    ids = {b.policy_id for b in budgets}
    assert "team-engineering-monthly" in ids
    assert buffer_pct == 0.15


# ── Complexity tier ceiling (with real registry) ──────────────────────────────

@pytest.mark.asyncio
async def test_critical_task_selects_tier1_only(selector, registry):
    """Critical complexity must only select tier-1 models."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        decision = await selector.select(_task(
            complexity=Complexity.critical,
            domain=Domain.reasoning,
        ))
    model = registry.get(decision.chosen_model_id)
    assert model.tier == ModelTier.premium, (
        f"Expected tier-1 for critical task, got {decision.chosen_model_id} (tier {model.tier})"
    )


@pytest.mark.asyncio
async def test_complex_task_uses_at_most_tier2(selector, registry):
    """Complex complexity allows tier ≤ 2; no tier-3 or tier-4 models."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        decision = await selector.select(_task(
            complexity=Complexity.complex,
            domain=Domain.reasoning,
        ))
    model = registry.get(decision.chosen_model_id)
    assert model.tier.value <= 2, (
        f"Expected tier ≤ 2 for complex task, got {decision.chosen_model_id} (tier {model.tier})"
    )


@pytest.mark.asyncio
async def test_moderate_task_excludes_tier4_local(selector, registry):
    """Moderate complexity allows tier ≤ 3; tier-4 local models must be excluded."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        decision = await selector.select(_task(
            complexity=Complexity.moderate,
            domain=Domain.chat,
        ))
    model = registry.get(decision.chosen_model_id)
    assert model.tier != ModelTier.local, (
        f"Tier-4 local model selected for moderate task: {decision.chosen_model_id}"
    )


@pytest.mark.asyncio
async def test_simple_task_selects_any_tier(selector, registry):
    """Simple tasks have no ceiling; the selector must return *something*."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        decision = await selector.select(_task(complexity=Complexity.simple))
    assert decision.accepted
    assert registry.get(decision.chosen_model_id) is not None


# ── Privacy filter ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_confidential_task_routes_to_local_only(selector, registry):
    """Privacy.confidential must select a local (is_local=True) model."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        decision = await selector.select(_task(
            complexity=Complexity.simple,
            domain=Domain.chat,
            privacy=Privacy.confidential,
        ))
    model = registry.get(decision.chosen_model_id)
    assert model.is_local, (
        f"Non-local model selected for confidential task: {decision.chosen_model_id}"
    )


# ── Budget enforcement ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_team_without_budget_policy_is_not_blocked(selector):
    """A team_id with no budget policy should always be allowed through."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        decision = await selector.select(_task(team_id="team-with-no-policy"))
    assert decision.accepted


@pytest.mark.asyncio
async def test_exhausted_budget_blocks_paid_models_routes_to_free(registry, guardrail_policy, buffer_pct):
    """Budget exhaustion blocks all paid cloud models; free local models (cost=$0) are exempt.

    This is correct: free local models never consume budget, so they always
    remain available regardless of the team's spending limit.
    """
    from tidus.models.budget import BudgetPolicy, BudgetPeriod, BudgetScope

    tiny_policy = BudgetPolicy(
        policy_id="tiny",
        scope=BudgetScope.team,
        scope_id="team-broke",
        period=BudgetPeriod.monthly,
        limit_usd=0.000001,  # $0.000001 — all paid models will exceed this
        hard_stop=True,
    )
    counter = SpendCounter()
    enforcer = BudgetEnforcer([tiny_policy], counter)
    matcher = CapabilityMatcher(guardrail_policy)
    engine = CostEngine(buffer_pct=buffer_pct)
    selector = ModelSelector(registry, enforcer, matcher, engine)

    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        decision = await selector.select(_task(
            team_id="team-broke",
            complexity=Complexity.simple,
            domain=Domain.chat,
        ))

    # Free local models are always available (cost=0.0 never exceeds any limit)
    model = registry.get(decision.chosen_model_id)
    assert model.is_local and model.input_price == 0.0, (
        f"Expected a free local model under budget exhaustion, got {decision.chosen_model_id}"
    )


@pytest.mark.asyncio
async def test_exhausted_budget_and_no_local_models_raises(registry, guardrail_policy, buffer_pct):
    """When budget is exhausted AND the task requires cloud-only models, raise ModelSelectionError."""
    from tidus.models.budget import BudgetPolicy, BudgetPeriod, BudgetScope

    tiny_policy = BudgetPolicy(
        policy_id="tiny-critical",
        scope=BudgetScope.team,
        scope_id="team-broke-critical",
        period=BudgetPeriod.monthly,
        limit_usd=0.000001,
        hard_stop=True,
    )
    counter = SpendCounter()
    enforcer = BudgetEnforcer([tiny_policy], counter)
    matcher = CapabilityMatcher(guardrail_policy)
    engine = CostEngine(buffer_pct=buffer_pct)
    selector = ModelSelector(registry, enforcer, matcher, engine)

    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        with pytest.raises(ModelSelectionError) as exc_info:
            # Critical tasks require tier-1 only; no tier-1 models are free
            await selector.select(_task(
                team_id="team-broke-critical",
                complexity=Complexity.critical,
                domain=Domain.reasoning,
            ))
    assert exc_info.value.stage == 4


# ── Preferred model override ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_preferred_model_respected_when_eligible(selector, registry):
    """When preferred_model_id names an eligible model, it must be selected."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        decision = await selector.select(_task(
            complexity=Complexity.simple,
            domain=Domain.chat,
            preferred_model_id="claude-haiku-4-5",
        ))
    assert decision.chosen_model_id == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_preferred_model_bypassed_when_ineligible(selector, registry):
    """A preferred tier-3 model should be bypassed for a critical task (tier-1 only)."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        decision = await selector.select(_task(
            complexity=Complexity.critical,
            domain=Domain.reasoning,
            preferred_model_id="claude-haiku-4-5",  # tier-3 — fails tier ceiling
        ))
    # Should still succeed but with a different model
    model = registry.get(decision.chosen_model_id)
    assert model.tier == ModelTier.premium
    assert decision.chosen_model_id != "claude-haiku-4-5"


# ── Guardrails ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_depth_5_is_allowed(selector):
    """Depth exactly at the limit (5) should still pass."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        decision = await selector.select(_task(agent_depth=5))
    assert decision.accepted


@pytest.mark.asyncio
async def test_agent_depth_6_is_rejected(selector):
    """Depth exceeding the limit (5) must raise ModelSelectionError."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        with pytest.raises(ModelSelectionError):
            await selector.select(_task(agent_depth=6))


# ── Scoring sanity ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_selected_model_supports_requested_domain(selector, registry):
    """The chosen model must actually support the task's domain."""
    with patch("tidus.cost.engine.count_tokens", new=AsyncMock(return_value=200)):
        for domain in (Domain.code, Domain.summarization, Domain.classification):
            decision = await selector.select(_task(domain=domain))
            model = registry.get(decision.chosen_model_id)
            cap_values = [c.value for c in model.capabilities]
            assert domain.value in cap_values, (
                f"Selected model {decision.chosen_model_id} doesn't support {domain}"
            )
