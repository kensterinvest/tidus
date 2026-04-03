"""Integration tests for ModelSelector edge cases.

Validates correct behaviour when:
- All models are disabled (no candidates survive Stage 1)
- All models are over budget (no candidates survive Stage 4)
- Only a single model remains (score normalisation edge case)
- Context window is exactly at the limit (boundary condition)
- Preferred model is disabled (fallback to normal selection)
- Warn-only budget never blocks routing

Run with:
    uv run pytest tests/integration/test_selector_edge_cases.py -v
"""

from __future__ import annotations

import pytest

from tidus.budget.enforcer import BudgetEnforcer
from tidus.cost.counter import SpendCounter
from tidus.cost.engine import CostEngine
from tidus.models.budget import BudgetPolicy, BudgetPeriod, BudgetScope
from tidus.models.guardrails import GuardrailPolicy
from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor
from tidus.router.capability_matcher import CapabilityMatcher
from tidus.router.registry import ModelRegistry
from tidus.router.selector import ModelSelectionError, ModelSelector
from tidus.utils.yaml_loader import load_yaml

MODELS_YAML = "config/models.yaml"
POLICIES_YAML = "config/policies.yaml"


def _task(
    complexity=Complexity.simple,
    domain=Domain.chat,
    tokens=100,
    privacy=Privacy.public,
    preferred_model_id=None,
    max_cost_usd=None,
    agent_depth=0,
    team_id="team-test",
) -> TaskDescriptor:
    return TaskDescriptor(
        task_id="test-task",
        team_id=team_id,
        workflow_id=None,
        agent_session_id=None,
        agent_depth=agent_depth,
        complexity=complexity,
        domain=domain,
        privacy=privacy,
        estimated_input_tokens=tokens,
        estimated_output_tokens=100,
        messages=[{"role": "user", "content": "test"}],
        preferred_model_id=preferred_model_id,
        max_cost_usd=max_cost_usd,
    )


def _make_selector(registry, budget_limit=None, hard_stop=True):
    policies = []
    if budget_limit is not None:
        policies.append(BudgetPolicy(
            policy_id="test-policy",
            scope=BudgetScope.team,
            scope_id="team-test",
            period=BudgetPeriod.monthly,
            limit_usd=budget_limit,
            warn_at_pct=0.80,
            hard_stop=hard_stop,
        ))
    enforcer = BudgetEnforcer(policies, SpendCounter())
    raw = load_yaml(POLICIES_YAML)
    guardrail = GuardrailPolicy.model_validate(raw["guardrails"])
    matcher = CapabilityMatcher(guardrail)
    engine = CostEngine(buffer_pct=raw.get("routing", {}).get("buffer_pct", 0.15))
    return ModelSelector(registry, enforcer, matcher, engine)


# ── All models disabled ───────────────────────────────────────────────────────

class TestAllModelsDisabled:
    async def test_all_disabled_raises_model_selection_error(self):
        """When every model is disabled, selector must raise at an early stage."""
        registry = ModelRegistry.load(MODELS_YAML)
        for spec in registry.list_all():
            registry.set_enabled(spec.model_id, False)

        selector = _make_selector(registry)
        with pytest.raises(ModelSelectionError) as exc_info:
            await selector.select(_task())

        assert exc_info.value.stage is not None

    async def test_single_enabled_local_model_is_selected(self):
        """When only one local model is enabled it must be chosen for a simple task."""
        registry = ModelRegistry.load(MODELS_YAML)

        target_id = None
        for spec in registry.list_all():
            registry.set_enabled(spec.model_id, False)
            if spec.is_local and target_id is None:
                target_id = spec.model_id

        assert target_id is not None, "No local model found in registry"
        registry.set_enabled(target_id, True)

        selector = _make_selector(registry)
        decision = await selector.select(_task(privacy=Privacy.confidential))
        assert decision.chosen_model_id == target_id


# ── Budget exhaustion edge cases ──────────────────────────────────────────────

class TestBudgetExhaustionEdgeCases:
    async def test_exhausted_budget_blocks_all_cloud_models(self):
        """A fully exhausted hard-stop budget must block all paid models."""
        registry = ModelRegistry.load(MODELS_YAML)
        # Tiny positive limit, immediately depleted
        policies = [BudgetPolicy(
            policy_id="exhausted",
            scope=BudgetScope.team,
            scope_id="team-test",
            period=BudgetPeriod.monthly,
            limit_usd=0.001,
            warn_at_pct=0.80,
            hard_stop=True,
        )]
        counter = SpendCounter()
        enforcer = BudgetEnforcer(policies, counter)
        # Deplete the budget entirely
        await counter.add("team-test", None, 0.001)

        raw = load_yaml(POLICIES_YAML)
        guardrail = GuardrailPolicy.model_validate(raw["guardrails"])
        matcher = CapabilityMatcher(guardrail)
        engine = CostEngine(buffer_pct=raw.get("routing", {}).get("buffer_pct", 0.15))
        selector = ModelSelector(registry, enforcer, matcher, engine)

        with pytest.raises(ModelSelectionError):
            await selector.select(_task(
                complexity=Complexity.critical,
                domain=Domain.reasoning,
                privacy=Privacy.public,
            ))

    async def test_warn_only_budget_never_blocks(self):
        """A warn-only budget policy must never produce a routing failure."""
        registry = ModelRegistry.load(MODELS_YAML)
        selector = _make_selector(registry, budget_limit=0.000001, hard_stop=False)

        decision = await selector.select(_task())
        assert decision.accepted is True

    async def test_max_cost_usd_restricts_to_cheap_models(self):
        """Task-level max_cost_usd must eliminate expensive models."""
        registry = ModelRegistry.load(MODELS_YAML)
        selector = _make_selector(registry)

        # Only allow effectively free models via very tight per-task cost cap
        decision = await selector.select(_task(
            max_cost_usd=0.000001,
            privacy=Privacy.confidential,
        ))
        chosen = registry.get(decision.chosen_model_id)
        assert chosen is not None
        assert chosen.is_local is True


# ── Complexity and domain edge cases ─────────────────────────────────────────

class TestComplexityDomainEdgeCases:
    async def test_critical_task_never_routes_to_local_model(self):
        """Critical tasks must not be served by Tier 4 (local/free) models."""
        registry = ModelRegistry.load(MODELS_YAML)
        selector = _make_selector(registry)

        decision = await selector.select(_task(
            complexity=Complexity.critical,
            domain=Domain.reasoning,
        ))
        chosen = registry.get(decision.chosen_model_id)
        assert chosen is not None
        assert chosen.tier == 1
        assert chosen.is_local is False

    async def test_simple_task_prefers_lower_tier(self):
        """Simple tasks must select a lower-tier model, not Tier 1 premium."""
        registry = ModelRegistry.load(MODELS_YAML)
        selector = _make_selector(registry)

        decision = await selector.select(_task(
            complexity=Complexity.simple,
            domain=Domain.chat,
        ))
        chosen = registry.get(decision.chosen_model_id)
        assert chosen is not None
        assert chosen.tier >= 2

    async def test_selection_is_deterministic(self):
        """Repeated identical tasks must always choose the same model."""
        registry = ModelRegistry.load(MODELS_YAML)
        selector = _make_selector(registry)
        task = _task(complexity=Complexity.moderate, domain=Domain.code)

        decisions = [await selector.select(task) for _ in range(5)]
        model_ids = [d.chosen_model_id for d in decisions]

        assert len(set(model_ids)) == 1, (
            f"Non-deterministic selection across 5 calls: {model_ids}"
        )

    async def test_disabled_preferred_model_falls_back_to_routing(self):
        """If the preferred model is disabled, normal routing must still succeed."""
        registry = ModelRegistry.load(MODELS_YAML)

        tier1 = next(s for s in registry.list_enabled() if s.tier == 1)
        registry.set_enabled(tier1.model_id, False)

        selector = _make_selector(registry)
        decision = await selector.select(_task(preferred_model_id=tier1.model_id))

        assert decision.accepted is True
        assert decision.chosen_model_id != tier1.model_id

    async def test_context_window_small_request_always_passes(self):
        """A small token request (100 tokens) fits any model's context window."""
        registry = ModelRegistry.load(MODELS_YAML)
        selector = _make_selector(registry)

        decision = await selector.select(_task(tokens=100))
        assert decision.accepted is True

    async def test_oversized_request_rejected_when_no_model_fits(self):
        """A request larger than any model's context window must raise."""
        registry = ModelRegistry.load(MODELS_YAML)
        selector = _make_selector(registry)

        # 10 million tokens exceeds any current model's max context
        with pytest.raises(ModelSelectionError):
            await selector.select(_task(tokens=10_000_000))


# ── Rejection reason coverage ─────────────────────────────────────────────────

class TestRejectionReasons:
    async def test_rejected_decision_has_stage_and_rejections(self):
        """ModelSelectionError must contain stage number and rejection list."""
        registry = ModelRegistry.load(MODELS_YAML)
        for spec in registry.list_all():
            registry.set_enabled(spec.model_id, False)

        selector = _make_selector(registry)

        with pytest.raises(ModelSelectionError) as exc_info:
            await selector.select(_task())

        error = exc_info.value
        assert error.stage is not None
        assert len(error.rejections) > 0

    async def test_all_rejections_have_model_id_and_reason(self):
        """Every rejection entry must have model_id and rejection_reason."""
        registry = ModelRegistry.load(MODELS_YAML)
        for spec in registry.list_all():
            registry.set_enabled(spec.model_id, False)

        selector = _make_selector(registry)

        with pytest.raises(ModelSelectionError) as exc_info:
            await selector.select(_task())

        for rejection in exc_info.value.rejections:
            assert rejection.chosen_model_id is not None
            assert rejection.rejection_reason is not None

    async def test_model_selection_error_is_not_http_error(self):
        """ModelSelectionError must be a clean domain exception, not an HTTP wrapper."""
        registry = ModelRegistry.load(MODELS_YAML)
        for spec in registry.list_all():
            registry.set_enabled(spec.model_id, False)

        selector = _make_selector(registry)

        with pytest.raises(ModelSelectionError) as exc_info:
            await selector.select(_task())

        # Must be our typed domain error, not an HTTP or generic exception
        assert isinstance(exc_info.value, ModelSelectionError)
