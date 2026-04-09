"""Model selector — 5-stage model selection algorithm.

Stage 1: Hard constraints     (CapabilityMatcher)
Stage 2: Guardrails           (CapabilityMatcher)
Stage 3: Complexity tier ceiling
Stage 4: Budget filter        (BudgetEnforcer)
Stage 5: Score & select       (cost × 0.70 + tier × 0.20 + latency × 0.10)

If no model survives all stages, raises ModelSelectionError with the stage
that eliminated every candidate and the reasons why.

Example:
    selector = ModelSelector(registry, enforcer, matcher, policies)
    decision = await selector.select(task)
"""

from __future__ import annotations

import math

import structlog

from tidus.budget.enforcer import BudgetEnforcer
from tidus.cost.engine import CostEngine
from tidus.models.model_registry import ModelSpec
from tidus.models.routing import RejectionReason, RoutingDecision
from tidus.models.task import Complexity, TaskDescriptor
from tidus.router.capability_matcher import CapabilityMatcher
from tidus.router.registry import ModelRegistry

log = structlog.get_logger(__name__)

# Complexity → maximum allowed tier (1=premium, 4=local/free)
_COMPLEXITY_TIER_CEILING: dict[Complexity, int] = {
    Complexity.simple: 4,
    Complexity.moderate: 3,
    Complexity.complex: 2,
    Complexity.critical: 1,
}


class ModelSelectionError(Exception):
    """Raised when no model survives all 5 selection stages."""

    def __init__(self, message: str, stage: int, rejections: list[RoutingDecision]) -> None:
        super().__init__(message)
        self.stage = stage
        self.rejections = rejections


class ModelSelector:
    """Orchestrates the 5-stage model selection pipeline."""

    def __init__(
        self,
        registry: ModelRegistry,
        enforcer: BudgetEnforcer,
        matcher: CapabilityMatcher,
        cost_engine: CostEngine,
    ) -> None:
        self._registry = registry
        self._enforcer = enforcer
        self._matcher = matcher
        self._cost_engine = cost_engine

    # ── Public API ────────────────────────────────────────────────────────────

    async def select(self, task: TaskDescriptor) -> RoutingDecision:
        """Run all 5 stages and return the best RoutingDecision.

        If a preferred_model_id is set on the task and it survives all stages,
        it is returned directly (bypasses scoring but still enforces budget).

        Raises ModelSelectionError if no model survives.
        """
        all_rejections: list[RoutingDecision] = []

        # ── Stage 1 & 2: Hard constraints + guardrails ─────────────────────
        # Use list_all() so disabled/deprecated models appear in rejection log
        candidates = self._registry.list_all()
        eligible, rejected = self._matcher.filter(candidates, task)
        all_rejections.extend(rejected)

        if not eligible:
            raise ModelSelectionError(
                f"No models passed hard constraints/guardrails for task {task.task_id}",
                stage=2,
                rejections=all_rejections,
            )

        # ── Stage 3: Complexity tier ceiling ───────────────────────────────
        tier_ceiling = _COMPLEXITY_TIER_CEILING[task.complexity]
        after_tier = [s for s in eligible if s.tier <= tier_ceiling]
        for spec in eligible:
            if spec.tier > tier_ceiling:
                all_rejections.append(
                    RoutingDecision(
                        task_id=task.task_id,
                        chosen_model_id=spec.model_id,
                        rejection_reason=RejectionReason.complexity_ceiling,
                        score=None,
                        estimated_cost_usd=None,
                    )
                )

        if not after_tier:
            raise ModelSelectionError(
                f"No models within tier ceiling {tier_ceiling} for complexity "
                f"'{task.complexity}' on task {task.task_id}",
                stage=3,
                rejections=all_rejections,
            )

        # ── Stage 4: Budget filter ─────────────────────────────────────────
        costed: list[tuple[ModelSpec, float]] = []
        for spec in after_tier:
            estimate = await self._cost_engine.estimate(spec, task)
            cost_usd = estimate.estimated_cost_usd

            # Honour per-request cost cap set by the caller
            if task.max_cost_usd is not None and cost_usd > task.max_cost_usd:
                all_rejections.append(
                    RoutingDecision(
                        task_id=task.task_id,
                        chosen_model_id=spec.model_id,
                        rejection_reason=RejectionReason.budget_exceeded,
                        score=None,
                        estimated_cost_usd=cost_usd,
                    )
                )
                continue

            can_spend = await self._enforcer.can_spend(
                team_id=task.team_id,
                workflow_id=task.workflow_id,
                amount_usd=cost_usd,
            )
            if not can_spend:
                all_rejections.append(
                    RoutingDecision(
                        task_id=task.task_id,
                        chosen_model_id=spec.model_id,
                        rejection_reason=RejectionReason.budget_exceeded,
                        score=None,
                        estimated_cost_usd=cost_usd,
                    )
                )
                continue

            costed.append((spec, cost_usd))

        if not costed:
            raise ModelSelectionError(
                f"All models exceed budget for task {task.task_id}",
                stage=4,
                rejections=all_rejections,
            )

        # ── Stage 5: Score & select ────────────────────────────────────────
        # If the caller pinned a preferred model and it made it through, use it.
        if task.preferred_model_id:
            for spec, cost_usd in costed:
                if spec.model_id == task.preferred_model_id:
                    log.info(
                        "preferred_model_selected",
                        task_id=task.task_id,
                        model_id=spec.model_id,
                        estimated_cost_usd=cost_usd,
                    )
                    return RoutingDecision(
                        task_id=task.task_id,
                        chosen_model_id=spec.model_id,
                        rejection_reason=None,
                        score=0.0,
                        estimated_cost_usd=cost_usd,
                    )

        best_spec, best_cost, best_score = _score_and_pick(costed)

        log.info(
            "model_selected",
            task_id=task.task_id,
            model_id=best_spec.model_id,
            score=round(best_score, 4),
            estimated_cost_usd=best_cost,
            candidates_passed=len(costed),
        )

        return RoutingDecision(
            task_id=task.task_id,
            chosen_model_id=best_spec.model_id,
            rejection_reason=None,
            score=best_score,
            estimated_cost_usd=best_cost,
        )


# ── Scoring ────────────────────────────────────────────────────────────────────

_DEPRECATED_SCORE_PENALTY = 0.15  # added to normalised score when model is deprecated

# Penalty is intentionally modest: deprecated models should lose to equally-priced
# non-deprecated models but still win if they are significantly cheaper or faster.
# Logged as routing_deprecated_model when a deprecated model is selected.


def _score_and_pick(
    costed: list[tuple[ModelSpec, float]],
) -> tuple[ModelSpec, float, float]:
    """Score each candidate and return (best_spec, best_cost, best_score).

    Score = cost_norm×0.70 + tier_norm×0.20 + latency_norm×0.10 [+ deprecated_penalty]
    Lower score = better (minimum cost wins).

    Each dimension is normalised to [0, 1] across the candidate set so that
    the weights have consistent meaning regardless of the units involved.
    A single-candidate set always gets score 0.0.
    Deprecated models receive a flat 0.15 penalty added after normalisation.
    """
    if len(costed) == 1:
        spec, cost = costed[0]
        if spec.deprecated:
            log.warning("routing_deprecated_model", model_id=spec.model_id)
        return spec, cost, 0.0

    costs = [c for _, c in costed]
    tiers = [float(s.tier) for s, _ in costed]
    latencies = [float(s.latency_p50_ms) for s, _ in costed]

    cost_norm = _normalize(costs)
    tier_norm = _normalize(tiers)
    lat_norm = _normalize(latencies)

    best_score = math.inf
    best_spec: ModelSpec = costed[0][0]
    best_cost: float = costed[0][1]

    for i, (spec, cost) in enumerate(costed):
        score = cost_norm[i] * 0.70 + tier_norm[i] * 0.20 + lat_norm[i] * 0.10
        if spec.deprecated:
            score += _DEPRECATED_SCORE_PENALTY
        if score < best_score:
            best_score = score
            best_spec = spec
            best_cost = cost

    if best_spec.deprecated:
        log.warning("routing_deprecated_model", model_id=best_spec.model_id, score=round(best_score, 4))

    return best_spec, best_cost, best_score


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalise a list to [0, 1]. All-equal values → all 0.0."""
    lo, hi = min(values), max(values)
    spread = hi - lo
    if spread == 0:
        return [0.0] * len(values)
    return [(v - lo) / spread for v in values]
