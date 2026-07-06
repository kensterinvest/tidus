"""Integration regression — a Claude-discovered ("dark") model must not be
selectable through the real model registry until claude_discovery_routing_enabled
is flipped on.

This mirrors tests/integration/test_selector_real_registry.py (real config/models.yaml
registry, real GuardrailPolicy) but pools in one synthetic route_source="claude_market"
spec to prove the dark-gate holds end-to-end against the production registry, not
just the small in-memory fixtures used by the unit test in
tests/unit/router/test_capability_matcher_claude_dark.py.
"""

from __future__ import annotations

from datetime import date

import pytest

from tidus.models.guardrails import GuardrailPolicy
from tidus.models.model_registry import Capability, ModelSpec, ModelTier, TokenizerType
from tidus.models.routing import RejectionReason
from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor
from tidus.router.capability_matcher import CapabilityMatcher
from tidus.router.registry import ModelRegistry
from tidus.utils.yaml_loader import load_yaml

MODELS_YAML = "config/models.yaml"
POLICIES_YAML = "config/policies.yaml"


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry.load(MODELS_YAML)


@pytest.fixture(scope="module")
def guardrail_policy():
    raw = load_yaml(POLICIES_YAML)
    return GuardrailPolicy.model_validate(raw["guardrails"])


def _dark_spec() -> ModelSpec:
    """A Claude-discovered flagship that would otherwise be eligible for a
    COMPLEX task — critical-capable, tier-1, broad complexity range."""
    return ModelSpec(
        model_id="acme-ultra-2",
        vendor="acme",
        tier=ModelTier.premium,
        max_context=200000,
        input_price=0.005,
        output_price=0.02,
        tokenizer=TokenizerType.tiktoken_cl100k,
        capabilities=[Capability.chat, Capability.code, Capability.reasoning],
        min_complexity="simple",
        max_complexity="critical",
        fallbacks=[],
        last_price_check=date(2025, 1, 1),
        route_source="claude_market",
    )


def _complex_task() -> TaskDescriptor:
    return TaskDescriptor(
        team_id="team-eng",
        complexity=Complexity.complex,
        domain=Domain.reasoning,
        privacy=Privacy.public,
        estimated_input_tokens=500,
        messages=[{"role": "user", "content": "solve this"}],
    )


def test_dark_model_rejected_against_real_registry_when_flag_off(registry, guardrail_policy):
    """With the flag off, the dark model is rejected alongside — not instead of —
    the real registry's own eligible candidates for a COMPLEX task."""
    matcher = CapabilityMatcher(guardrail_policy, claude_discovery_routing_enabled=False)
    pool = [*registry.list_all(), _dark_spec()]
    eligible, rejected = matcher.filter(pool, _complex_task())

    assert "acme-ultra-2" not in [s.model_id for s in eligible]
    # Real, non-dark models remain eligible — the gate targets only route_source="claude_market".
    assert any(s.route_source != "claude_market" for s in eligible)
    dark_rejections = [r for r in rejected if r.chosen_model_id == "acme-ultra-2"]
    assert len(dark_rejections) == 1
    assert dark_rejections[0].rejection_reason == RejectionReason.claude_discovery_routing_disabled


def test_dark_model_eligible_against_real_registry_when_flag_on(registry, guardrail_policy):
    """Flipping the flag on admits the dark model as a genuine candidate
    alongside the real registry's eligible models for a COMPLEX task."""
    matcher = CapabilityMatcher(guardrail_policy, claude_discovery_routing_enabled=True)
    pool = [*registry.list_all(), _dark_spec()]
    eligible, _ = matcher.filter(pool, _complex_task())

    assert "acme-ultra-2" in [s.model_id for s in eligible]
