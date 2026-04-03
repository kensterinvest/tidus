"""Budget enforcer — can_spend() check and deduct() after successful execution.

The enforcer holds all active BudgetPolicy objects and a SpendCounter. On each
routing request it checks whether the estimated cost fits within every
applicable policy before authorising spend. After execution the adapter calls
deduct() to commit the actual cost.

Warning thresholds are evaluated on every deduction so that a single log line
(and future alert hook) fires when a team crosses 80% of their monthly budget.

Example:
    enforcer = BudgetEnforcer(policies, counter)
    ok = await enforcer.can_spend("team-eng", "wf-chat", 0.002)
    if ok:
        await enforcer.deduct("team-eng", "wf-chat", actual_cost)
"""

from __future__ import annotations

import structlog

from tidus.cost.counter import SpendCounter
from tidus.models.budget import BudgetPolicy, BudgetScope, BudgetStatus

log = structlog.get_logger(__name__)


class BudgetEnforcer:
    """Checks and records spend against team and workflow budget policies."""

    def __init__(
        self,
        policies: list[BudgetPolicy],
        counter: SpendCounter,
    ) -> None:
        self._policies = policies
        self._counter = counter

    # ── Policy management ─────────────────────────────────────────────────────

    def list_policies(self) -> list[BudgetPolicy]:
        """Return all active budget policies."""
        return list(self._policies)

    def add_policy(self, policy: BudgetPolicy) -> None:
        """Add a new budget policy (takes effect immediately)."""
        self._policies.append(policy)

    # ── Policy lookup helpers ─────────────────────────────────────────────────

    def _team_policy(self, team_id: str) -> BudgetPolicy | None:
        for p in self._policies:
            if p.scope == BudgetScope.team and p.scope_id == team_id:
                return p
        return None

    def _workflow_policy(self, workflow_id: str) -> BudgetPolicy | None:
        for p in self._policies:
            if p.scope == BudgetScope.workflow and p.scope_id == workflow_id:
                return p
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    async def can_spend(
        self,
        team_id: str,
        workflow_id: str | None,
        amount_usd: float,
    ) -> bool:
        """Return True if the estimated spend fits all applicable budget policies.

        Checks the team-level policy first, then the workflow-level policy if
        one exists. Returns False as soon as any policy would be exceeded.

        For policies with hard_stop=False, always returns True (warn-only mode).

        Example:
            ok = await enforcer.can_spend("team-eng", None, 0.005)
        """
        team_policy = self._team_policy(team_id)
        if team_policy and team_policy.hard_stop:
            allowed, current = await self._counter.check_and_add(
                team_id, None, amount_usd, team_policy.limit_usd
            )
            if not allowed:
                log.warning(
                    "budget_hard_stop",
                    team_id=team_id,
                    current_usd=current,
                    requested_usd=amount_usd,
                    limit_usd=team_policy.limit_usd,
                )
                return False
            # Undo the tentative deduction — deduct() will commit the real cost later
            await self._counter.add(team_id, None, -amount_usd)

        if workflow_id:
            wf_policy = self._workflow_policy(workflow_id)
            if wf_policy and wf_policy.hard_stop:
                allowed, current = await self._counter.check_and_add(
                    team_id, workflow_id, amount_usd, wf_policy.limit_usd
                )
                if not allowed:
                    log.warning(
                        "workflow_budget_hard_stop",
                        workflow_id=workflow_id,
                        current_usd=current,
                        requested_usd=amount_usd,
                        limit_usd=wf_policy.limit_usd,
                    )
                    return False
                await self._counter.add(team_id, workflow_id, -amount_usd)

        return True

    async def deduct(
        self,
        team_id: str,
        workflow_id: str | None,
        amount_usd: float,
    ) -> None:
        """Commit actual spend after a successful model call.

        Updates both the team-level and (if applicable) workflow-level counters.
        Emits a warning log if any policy's warn_at_pct threshold is crossed.

        Example:
            await enforcer.deduct("team-eng", "wf-chat", 0.0018)
        """
        new_team_total = await self._counter.add(team_id, None, amount_usd)

        team_policy = self._team_policy(team_id)
        if team_policy:
            utilisation = new_team_total / team_policy.limit_usd
            if utilisation >= team_policy.warn_at_pct:
                log.warning(
                    "budget_warn_threshold_crossed",
                    team_id=team_id,
                    utilisation_pct=round(utilisation * 100, 1),
                    current_usd=new_team_total,
                    limit_usd=team_policy.limit_usd,
                )

        if workflow_id:
            new_wf_total = await self._counter.add(team_id, workflow_id, amount_usd)
            wf_policy = self._workflow_policy(workflow_id)
            if wf_policy:
                utilisation = new_wf_total / wf_policy.limit_usd
                if utilisation >= wf_policy.warn_at_pct:
                    log.warning(
                        "workflow_budget_warn_threshold_crossed",
                        workflow_id=workflow_id,
                        utilisation_pct=round(utilisation * 100, 1),
                        current_usd=new_wf_total,
                        limit_usd=wf_policy.limit_usd,
                    )

    async def status(self, team_id: str, workflow_id: str | None = None) -> BudgetStatus:
        """Return the current BudgetStatus for a team (and optionally workflow).

        Example:
            status = await enforcer.status("team-engineering")
        """
        policy = self._team_policy(team_id)
        if workflow_id:
            policy = self._workflow_policy(workflow_id) or policy

        spent = await self._counter.get(team_id, workflow_id)

        if policy is None:
            return BudgetStatus(
                policy_id="none",
                scope_id=team_id,
                limit_usd=0.0,
                spent_usd=spent,
                remaining_usd=0.0,
                utilisation_pct=0.0,
                is_over_warn_threshold=False,
                is_hard_stopped=False,
            )

        remaining = max(0.0, policy.limit_usd - spent)
        utilisation = spent / policy.limit_usd if policy.limit_usd > 0 else 0.0

        return BudgetStatus(
            policy_id=policy.policy_id,
            scope_id=policy.scope_id,
            limit_usd=policy.limit_usd,
            spent_usd=spent,
            remaining_usd=remaining,
            utilisation_pct=round(utilisation * 100, 2),
            is_over_warn_threshold=utilisation >= policy.warn_at_pct,
            is_hard_stopped=policy.hard_stop and spent >= policy.limit_usd,
        )
