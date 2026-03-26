"""Capability matcher — pure eligibility filter for model selection.

This module answers one question per model: "Is this model eligible for
this task?" It applies only hard, binary constraints — no scoring.

Stages implemented here (stages 1–2 of the 5-stage selector):
  1. Hard constraints — enabled, context fits, domain supported, privacy
  2. Guardrail constraints — agent depth, tokens-per-step, retry count

Scoring (stages 3–5) lives in selector.py.

Example:
    matcher = CapabilityMatcher(policies)
    eligible = matcher.filter(candidates, task)
"""

from tidus.models.guardrails import GuardrailPolicy
from tidus.models.model_registry import Capability, ModelSpec
from tidus.models.routing import RejectionReason, RoutingDecision
from tidus.models.task import Complexity, Domain, Privacy, TaskDescriptor

# Map Domain enum → Capability enum
_DOMAIN_TO_CAPABILITY: dict[Domain, Capability] = {
    Domain.chat: Capability.chat,
    Domain.code: Capability.code,
    Domain.reasoning: Capability.reasoning,
    Domain.extraction: Capability.extraction,
    Domain.classification: Capability.classification,
    Domain.summarization: Capability.summarization,
    Domain.creative: Capability.creative,
}

# Complexity ordering for tier ceiling checks
_COMPLEXITY_ORDER: dict[Complexity, int] = {
    Complexity.simple: 0,
    Complexity.moderate: 1,
    Complexity.complex: 2,
    Complexity.critical: 3,
}


class CapabilityMatcher:
    """Filters a list of ModelSpec objects against a TaskDescriptor.

    Each method applies one category of constraints and returns only the
    models that pass, plus a list of per-model rejection reasons for
    observability.
    """

    def __init__(self, guardrails: GuardrailPolicy) -> None:
        self._guardrails = guardrails

    # ── Public API ────────────────────────────────────────────────────────────

    def filter(
        self,
        candidates: list[ModelSpec],
        task: TaskDescriptor,
    ) -> tuple[list[ModelSpec], list[RoutingDecision]]:
        """Apply all hard constraints and guardrails.

        Returns:
            (eligible_models, rejected_decisions)
            where rejected_decisions contains one RoutingDecision per
            rejected model with the reason and is_rejected=True.
        """
        eligible: list[ModelSpec] = []
        rejected: list[RoutingDecision] = []

        for spec in candidates:
            reason = self._check_hard_constraints(spec, task)
            if reason is None:
                reason = self._check_guardrails(spec, task)

            if reason is None:
                eligible.append(spec)
            else:
                rejected.append(
                    RoutingDecision(
                        task_id=task.task_id,
                        chosen_model_id=spec.model_id,
                        rejection_reason=reason,
                        score=None,
                        estimated_cost_usd=None,
                    )
                )

        return eligible, rejected

    # ── Hard constraint checks ────────────────────────────────────────────────

    def _check_hard_constraints(
        self, spec: ModelSpec, task: TaskDescriptor
    ) -> RejectionReason | None:
        """Return a RejectionReason if the model fails any hard constraint."""

        # Must be enabled and not deprecated
        if not spec.enabled or spec.deprecated:
            return RejectionReason.model_disabled

        # Context window must fit the estimated input tokens
        if task.estimated_input_tokens > spec.max_context:
            return RejectionReason.context_too_large

        # Domain capability must be supported
        required_capability = _DOMAIN_TO_CAPABILITY.get(task.domain)
        if required_capability and required_capability not in spec.capabilities:
            return RejectionReason.domain_not_supported

        # Confidential tasks must stay on local/on-prem models
        if task.privacy == Privacy.confidential and not spec.is_local:
            return RejectionReason.privacy_violation

        # Task complexity must fall within the model's designed operating range.
        # This prevents routing a "simple" task to a model built for complex
        # reasoning (wasteful / wrong tool) and catches max_complexity violations
        # explicitly rather than relying solely on the tier ceiling.
        task_order = _COMPLEXITY_ORDER[task.complexity]
        model_min = _COMPLEXITY_ORDER.get(Complexity(spec.min_complexity), 0)
        model_max = _COMPLEXITY_ORDER.get(Complexity(spec.max_complexity), 3)
        if task_order < model_min or task_order > model_max:
            return RejectionReason.complexity_mismatch

        return None

    # ── Guardrail checks ──────────────────────────────────────────────────────

    def _check_guardrails(
        self, spec: ModelSpec, task: TaskDescriptor
    ) -> RejectionReason | None:
        """Return a RejectionReason if the task violates any guardrail policy."""

        if task.agent_depth > self._guardrails.max_agent_depth:
            return RejectionReason.agent_depth_exceeded

        if task.estimated_input_tokens > self._guardrails.max_tokens_per_step:
            return RejectionReason.token_limit_exceeded

        return None
