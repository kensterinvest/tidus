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
from tidus.models.budget import BudgetPeriod, BudgetPolicy, BudgetScope, BudgetStatus

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
    #
    # Budget enforcement uses a **reservation pattern** to avoid the
    # check-then-commit race:
    #
    #     can_spend(amount)      — pure read: would this fit right now?
    #     reserve(amount) → ok   — atomic check + hold; on True, the amount is
    #                              counted against the budget until refund/deduct.
    #     deduct(actual, reserved_usd=estimated)
    #                            — settle the reservation at the actual cost;
    #                              adjusts the counter by (actual - estimated).
    #     refund(amount)         — release a reservation when the adapter failed.
    #
    # `deduct(amount)` without `reserved_usd` preserves the older "add actual
    # directly" behavior used by tests and legacy callers.

    async def can_spend(
        self,
        team_id: str,
        workflow_id: str | None,
        amount_usd: float,
    ) -> bool:
        """Pure check — does ``amount_usd`` fit within all hard-stop policies?

        Reads current counters and compares against limits without mutating
        state. Used by the selector Stage 4 filter when iterating candidate
        models; the actual enforcement happens at :meth:`reserve`.

        Example:
            ok = await enforcer.can_spend("team-eng", None, 0.005)
        """
        team_policy = self._team_policy(team_id)
        if team_policy and team_policy.hard_stop:
            current = await self._counter.get(team_id, None)
            if current + amount_usd > team_policy.limit_usd:
                return False
        if workflow_id:
            wf_policy = self._workflow_policy(workflow_id)
            if wf_policy and wf_policy.hard_stop:
                current = await self._counter.get(team_id, workflow_id)
                if current + amount_usd > wf_policy.limit_usd:
                    return False
        return True

    async def reserve(
        self,
        team_id: str,
        workflow_id: str | None,
        amount_usd: float,
    ) -> bool:
        """Atomically check all hard-stop policies AND reserve ``amount_usd``.

        Returns True when every applicable hard-stop policy admitted the
        reservation; the amount is held on the counter until :meth:`deduct`
        (with ``reserved_usd``) or :meth:`refund` completes the lifecycle.

        If reservation fails for any scope, any partial reservation for other
        scopes is released before returning False.

        Example:
            if not await enforcer.reserve("team-eng", "wf-chat", 0.01):
                raise HTTPException(402, "Budget exceeded")
        """
        team_policy = self._team_policy(team_id)
        reserved_team = False

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
            reserved_team = True

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
                    if reserved_team:
                        # roll back the team-level reservation
                        await self._counter.add(team_id, None, -amount_usd)
                    return False

        return True

    async def refund(
        self,
        team_id: str,
        workflow_id: str | None,
        amount_usd: float,
    ) -> None:
        """Release a reservation made by :meth:`reserve`.

        Only scopes with hard-stop policies were reserved by ``reserve()``;
        we mirror that contract here so the counter ends in the same state
        it was in before the reservation.

        Example:
            await enforcer.refund("team-eng", "wf-chat", 0.01)
        """
        team_policy = self._team_policy(team_id)
        if team_policy and team_policy.hard_stop:
            await self._counter.add(team_id, None, -amount_usd)
        if workflow_id:
            wf_policy = self._workflow_policy(workflow_id)
            if wf_policy and wf_policy.hard_stop:
                await self._counter.add(team_id, workflow_id, -amount_usd)

    async def deduct(
        self,
        team_id: str,
        workflow_id: str | None,
        amount_usd: float,
        *,
        reserved_usd: float | None = None,
    ) -> None:
        """Commit actual spend after a successful model call.

        If ``reserved_usd`` is given the counter was already credited with the
        reservation, so we adjust by the delta ``amount_usd - reserved_usd``.
        Without ``reserved_usd`` the full ``amount_usd`` is added (legacy
        path used by tests and paths that did not call :meth:`reserve`).

        Warning thresholds evaluated against the post-adjustment total.

        Example:
            await enforcer.deduct("team-eng", "wf-chat", 0.0048, reserved_usd=0.005)
        """
        if reserved_usd is None:
            delta = amount_usd
        else:
            delta = amount_usd - reserved_usd

        new_team_total = await self._counter.add(team_id, None, delta)

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
            new_wf_total = await self._counter.add(team_id, workflow_id, delta)
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

    async def reset_period(self, period: BudgetPeriod) -> int:
        """Reset spend counters for all policies whose period matches *period*.

        Called by the monthly scheduler job on the 1st of each month so that
        teams start each billing period with a clean slate.

        Returns:
            The number of counters that were reset.
        """
        target_period = BudgetPeriod(period) if isinstance(period, str) else period
        count = 0
        for policy in self._policies:
            if policy.period != target_period:
                continue
            if policy.scope == BudgetScope.workflow:
                # Workflow counters are keyed (team_id, workflow_id); reset
                # across all teams that used this workflow.
                await self._counter.reset_workflow(policy.scope_id)
            else:
                await self._counter.reset(policy.scope_id, None)
            log.info(
                "budget_period_reset",
                policy_id=policy.policy_id,
                scope=policy.scope.value,
                scope_id=policy.scope_id,
                period=str(period),
            )
            count += 1
        if count:
            log.info("budget_period_reset_complete", reset_count=count, period=str(period))
        return count

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
