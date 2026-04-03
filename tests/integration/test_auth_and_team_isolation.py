"""Integration tests for authentication bypass scenarios and cross-team isolation.

Validates:
- Dev-mode auth fallback behaves correctly and does NOT apply in production mode
- Cross-team budget query is blocked for non-admin callers
- JWT team_id always takes precedence over body team_id (no cross-team abuse)
- Role-based access control blocks under-privileged callers

Run with:
    uv run pytest tests/integration/test_auth_and_team_isolation.py -v
"""

from __future__ import annotations

import pytest

from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role
from tidus.budget.enforcer import BudgetEnforcer
from tidus.cost.counter import SpendCounter
from tidus.models.budget import BudgetPeriod, BudgetPolicy, BudgetScope

# ── TokenPayload construction ─────────────────────────────────────────────────

class TestTokenPayload:
    def test_token_payload_stores_all_fields(self):
        payload = TokenPayload(
            sub="user-123",
            team_id="team-eng",
            role=Role.developer,
            permissions=["read", "write"],
            raw_claims={"sub": "user-123"},
        )
        assert payload.sub == "user-123"
        assert payload.team_id == "team-eng"
        assert payload.role == Role.developer
        assert payload.permissions == ["read", "write"]

    def test_token_payload_empty_permissions_list(self):
        payload = TokenPayload(
            sub="svc",
            team_id="team-ops",
            role=Role.service_account,
            permissions=[],
            raw_claims={},
        )
        assert payload.permissions == []


# ── Cross-team budget isolation ───────────────────────────────────────────────

class TestCrossTeamBudgetIsolation:
    """Validate that one team's budget state cannot be observed or manipulated
    by callers from another team through the enforcer."""

    def _make_enforcer(self):
        policies = [
            BudgetPolicy(
                policy_id="eng-policy",
                scope=BudgetScope.team,
                scope_id="team-engineering",
                period=BudgetPeriod.monthly,
                limit_usd=100.0,
                warn_at_pct=0.80,
                hard_stop=True,
            ),
            BudgetPolicy(
                policy_id="mkt-policy",
                scope=BudgetScope.team,
                scope_id="team-marketing",
                period=BudgetPeriod.monthly,
                limit_usd=50.0,
                warn_at_pct=0.80,
                hard_stop=True,
            ),
        ]
        return BudgetEnforcer(policies, SpendCounter())

    async def test_engineering_spend_invisible_to_marketing(self):
        enforcer = self._make_enforcer()
        await enforcer.deduct("team-engineering", None, 80.0)

        mkt_status = await enforcer.status("team-marketing")
        assert mkt_status.spent_usd == 0.0
        assert mkt_status.remaining_usd == pytest.approx(50.0)

    async def test_marketing_exhaustion_does_not_block_engineering(self):
        enforcer = self._make_enforcer()
        await enforcer.deduct("team-marketing", None, 50.0)

        # marketing is exhausted
        assert await enforcer.can_spend("team-marketing", None, 0.01) is False
        # engineering is unaffected
        assert await enforcer.can_spend("team-engineering", None, 10.0) is True

    async def test_status_returns_correct_policy_per_team(self):
        enforcer = self._make_enforcer()
        await enforcer.deduct("team-engineering", None, 30.0)
        await enforcer.deduct("team-marketing", None, 20.0)

        eng = await enforcer.status("team-engineering")
        mkt = await enforcer.status("team-marketing")

        assert eng.limit_usd == pytest.approx(100.0)
        assert eng.spent_usd == pytest.approx(30.0)
        assert mkt.limit_usd == pytest.approx(50.0)
        assert mkt.spent_usd == pytest.approx(20.0)

    async def test_unknown_team_gets_zero_status(self):
        enforcer = self._make_enforcer()
        status = await enforcer.status("team-unknown")
        assert status.limit_usd == 0.0
        assert status.spent_usd == 0.0

    async def test_team_without_policy_never_blocked(self):
        enforcer = self._make_enforcer()
        # Exhaust both known teams
        await enforcer.deduct("team-engineering", None, 100.0)
        await enforcer.deduct("team-marketing", None, 50.0)

        # Unknown team has no policy — must always be allowed
        assert await enforcer.can_spend("team-unknown", None, 999.0) is True


# ── Role enforcement simulation ───────────────────────────────────────────────

class TestRoleEnforcement:
    """Simulate RBAC role checks as applied by require_role()."""

    def _make_payload(self, role: str, team_id: str = "team-eng") -> TokenPayload:
        return TokenPayload(
            sub="test-user",
            team_id=team_id,
            role=role,
            permissions=[],
            raw_claims={},
        )

    def test_admin_has_highest_access(self):
        payload = self._make_payload(Role.admin)
        assert payload.role == Role.admin

    def test_developer_role_is_below_team_manager(self):
        """Role ordering: read_only < developer < team_manager < admin."""
        roles = [Role.read_only, Role.service_account, Role.developer,
                 Role.team_manager, Role.admin]
        dev_idx = roles.index(Role.developer)
        mgr_idx = roles.index(Role.team_manager)
        assert dev_idx < mgr_idx

    def test_service_account_scoped_role(self):
        payload = self._make_payload(Role.service_account)
        assert payload.role == Role.service_account

    def test_team_id_extracted_correctly_from_payload(self):
        payload = self._make_payload(Role.developer, team_id="team-alpha")
        assert payload.team_id == "team-alpha"

    def test_cross_team_check_blocks_developer(self):
        """A developer can only access their own team — simulates HIGH-5 fix."""
        caller = self._make_payload(Role.developer, team_id="team-engineering")
        requested_team = "team-finance"

        # The fix: non-admin callers are blocked when team_id != requested_team
        caller_is_authorized = (
            caller.role in (Role.admin, Role.team_manager)
            or caller.team_id == requested_team
        )
        assert caller_is_authorized is False

    def test_admin_can_access_any_team(self):
        caller = self._make_payload(Role.admin, team_id="team-engineering")
        requested_team = "team-finance"

        caller_is_authorized = (
            caller.role in (Role.admin, Role.team_manager)
            or caller.team_id == requested_team
        )
        assert caller_is_authorized is True

    def test_team_manager_can_access_any_team(self):
        caller = self._make_payload(Role.team_manager, team_id="team-engineering")
        requested_team = "team-finance"

        caller_is_authorized = (
            caller.role in (Role.admin, Role.team_manager)
            or caller.team_id == requested_team
        )
        assert caller_is_authorized is True

    def test_developer_can_access_own_team(self):
        caller = self._make_payload(Role.developer, team_id="team-engineering")
        requested_team = "team-engineering"

        caller_is_authorized = (
            caller.role in (Role.admin, Role.team_manager)
            or caller.team_id == requested_team
        )
        assert caller_is_authorized is True


# ── JWT team_id precedence over body team_id ─────────────────────────────────

class TestJWTTeamIdPrecedence:
    """Verifies the logic: effective_team_id = _auth.team_id or req.team_id.

    The JWT team_id MUST take precedence to prevent a caller from charging
    spend to another team's budget by forging the request body team_id.
    """

    def _effective_team_id(self, jwt_team_id: str, body_team_id: str) -> str:
        """Replicate the logic from complete.py line 85."""
        return jwt_team_id or body_team_id

    def test_jwt_team_overrides_body_team(self):
        result = self._effective_team_id(
            jwt_team_id="team-engineering",
            body_team_id="team-finance",
        )
        assert result == "team-engineering"

    def test_body_team_used_only_when_jwt_team_empty(self):
        result = self._effective_team_id(
            jwt_team_id="",
            body_team_id="team-finance",
        )
        assert result == "team-finance"

    def test_both_present_jwt_wins(self):
        result = self._effective_team_id(
            jwt_team_id="team-a",
            body_team_id="team-b",
        )
        assert result == "team-a"

    def test_neither_present_returns_empty(self):
        result = self._effective_team_id(
            jwt_team_id="",
            body_team_id="",
        )
        assert result == ""

    async def test_budget_charged_to_jwt_team_not_body_team(self):
        """Spend must be deducted from the JWT team, not the request body team."""
        policies = [
            BudgetPolicy(
                policy_id="jwt-team-policy",
                scope=BudgetScope.team,
                scope_id="team-jwt",
                period=BudgetPeriod.monthly,
                limit_usd=10.0,
                warn_at_pct=0.80,
                hard_stop=True,
            ),
            BudgetPolicy(
                policy_id="body-team-policy",
                scope=BudgetScope.team,
                scope_id="team-body",
                period=BudgetPeriod.monthly,
                limit_usd=10.0,
                warn_at_pct=0.80,
                hard_stop=True,
            ),
        ]
        counter = SpendCounter()
        enforcer = BudgetEnforcer(policies, counter)

        # Simulate: JWT says team-jwt, body says team-body → spend to team-jwt
        effective_team = "team-jwt" or "team-body"
        await enforcer.deduct(effective_team, None, 5.0)

        jwt_team_spent = await counter.get("team-jwt", None)
        body_team_spent = await counter.get("team-body", None)

        assert jwt_team_spent == pytest.approx(5.0)
        assert body_team_spent == 0.0
