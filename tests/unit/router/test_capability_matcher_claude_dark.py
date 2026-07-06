"""Unit tests for the Claude-discovery dark-gate (route_source="claude_market").

Mirrors the existing OpenRouter routability-flag tests in test_selector.py
(test_openrouter_model_rejected_when_routing_disabled et al.) but for the
route_source gate instead of the route_id gate.
"""

from __future__ import annotations

from datetime import date

from tidus.models.guardrails import GuardrailPolicy
from tidus.models.model_registry import Capability, ModelSpec, ModelTier, TokenizerType
from tidus.models.routing import RejectionReason
from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor
from tidus.router.capability_matcher import CapabilityMatcher


def _make_spec(model_id: str = "new-flagship", route_source: str | None = None) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        vendor="acme",
        tier=ModelTier.mid,
        max_context=200000,
        input_price=0.001,
        output_price=0.002,
        tokenizer=TokenizerType.tiktoken_cl100k,
        capabilities=[Capability.chat],
        min_complexity="simple",
        max_complexity="critical",
        fallbacks=[],
        last_price_check=date(2025, 1, 1),
        route_source=route_source,
    )


def _make_task() -> TaskDescriptor:
    return TaskDescriptor(
        team_id="team-eng",
        complexity=Complexity.simple,
        domain=Domain.chat,
        privacy=Privacy.public,
        estimated_input_tokens=10,
        messages=[{"role": "user", "content": "hello"}],
    )


def test_claude_market_model_rejected_when_flag_off():
    matcher = CapabilityMatcher(GuardrailPolicy(), claude_discovery_routing_enabled=False)
    spec = _make_spec(route_source="claude_market")
    eligible, rejected = matcher.filter([spec], _make_task())
    assert eligible == []
    assert any(r.rejection_reason == RejectionReason.claude_discovery_routing_disabled for r in rejected)


def test_claude_market_model_allowed_when_flag_on():
    matcher = CapabilityMatcher(GuardrailPolicy(), claude_discovery_routing_enabled=True)
    spec = _make_spec(route_source="claude_market")
    eligible, _ = matcher.filter([spec], _make_task())
    assert [s.model_id for s in eligible] == ["new-flagship"]


def test_normal_model_unaffected_by_flag():
    matcher = CapabilityMatcher(GuardrailPolicy(), claude_discovery_routing_enabled=False)
    spec = _make_spec(route_source=None)
    eligible, _ = matcher.filter([spec], _make_task())
    assert [s.model_id for s in eligible] == ["new-flagship"]
