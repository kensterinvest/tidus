"""Integration tests for Fix 4 — cross-team API isolation on 4 endpoints.

Regression tests for leaks flagged in the Opus 4.7 code review:
- `GET /api/v1/dashboard/summary`: no team filter on cost records / budgets / sessions
- `POST /api/v1/guardrails/sessions`: accepts team_id from body without auth match
- `GET/DELETE /api/v1/guardrails/sessions/{id}`: no team_id verification
- `POST /api/v1/budgets`: accepts scope_id from body without auth match
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tidus.auth.middleware import TokenPayload, get_current_user
from tidus.auth.rbac import Role
from tidus.main import create_app


def _payload(role: Role, team_id: str = "team-alpha") -> TokenPayload:
    return TokenPayload(
        sub="user-1",
        team_id=team_id,
        role=role.value,
        permissions=[],
        raw_claims={},
    )


@pytest.fixture(scope="module")
def test_app():
    """Module-scoped app; lifespan runs once so singletons exist."""
    app = create_app()
    with TestClient(app):
        yield app


@pytest.fixture
def as_developer(test_app):
    """TestClient authenticating as developer on team-alpha."""
    test_app.dependency_overrides[get_current_user] = lambda: _payload(
        Role.developer, "team-alpha"
    )
    try:
        yield TestClient(test_app)
    finally:
        test_app.dependency_overrides.clear()


@pytest.fixture
def as_admin(test_app):
    """TestClient authenticating as admin on team-admin."""
    test_app.dependency_overrides[get_current_user] = lambda: _payload(
        Role.admin, "team-admin"
    )
    try:
        yield TestClient(test_app)
    finally:
        test_app.dependency_overrides.clear()


def _with_role(test_app, role: Role, team_id: str) -> TestClient:
    """Helper for tests that need ad-hoc role/team combinations."""
    test_app.dependency_overrides[get_current_user] = lambda: _payload(role, team_id)
    return TestClient(test_app)


# ── Fix 4a: dashboard summary must not expose other-team data ─────────────────

class TestDashboardTeamFilter:
    def test_developer_sees_only_own_team_budget_rows(self, as_developer):
        """A developer in team-alpha must not see other teams' budget policies."""
        resp = as_developer.get("/api/v1/dashboard/summary")
        assert resp.status_code == 200, resp.text
        for row in resp.json()["budgets"]:
            assert row["team_id"] == "team-alpha", (
                f"Non-admin leaked other team's budget: {row['team_id']}"
            )

    def test_admin_sees_all_teams(self, as_admin):
        """Admin view is unrestricted."""
        resp = as_admin.get("/api/v1/dashboard/summary")
        assert resp.status_code == 200, resp.text
        assert isinstance(resp.json()["budgets"], list)


# ── Fix 4b: create_session must not accept other-team body.team_id ────────────

class TestCreateSessionTeamScope:
    def test_developer_cannot_create_session_for_other_team(self, as_developer):
        resp = as_developer.post(
            "/api/v1/guardrails/sessions",
            json={"session_id": "leak-attempt-1", "team_id": "team-beta"},
        )
        assert resp.status_code == 403, resp.text

    def test_developer_create_session_defaults_to_own_team(self, as_developer):
        resp = as_developer.post(
            "/api/v1/guardrails/sessions",
            json={"session_id": "own-session-1"},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["team_id"] == "team-alpha"
        as_developer.delete("/api/v1/guardrails/sessions/own-session-1")


# ── Fix 4c: get/terminate_session must verify team ownership ─────────────────

class TestSessionAccessTeamScope:
    def test_developer_cannot_read_other_team_session(
        self, test_app, as_admin, as_developer
    ):
        # Admin creates team-beta session
        admin_client = _with_role(test_app, Role.admin, "team-admin")
        create = admin_client.post(
            "/api/v1/guardrails/sessions",
            json={"session_id": "cross-team-read", "team_id": "team-beta"},
        )
        assert create.status_code == 201, create.text

        # Reset overrides for the developer client
        test_app.dependency_overrides[get_current_user] = lambda: _payload(
            Role.developer, "team-alpha"
        )
        try:
            resp = as_developer.get("/api/v1/guardrails/sessions/cross-team-read")
            assert resp.status_code == 404, (
                f"Team-beta session leaked to team-alpha developer: {resp.status_code}"
            )
        finally:
            test_app.dependency_overrides[get_current_user] = lambda: _payload(
                Role.admin, "team-admin"
            )
            admin_client.delete("/api/v1/guardrails/sessions/cross-team-read")

    def test_developer_cannot_terminate_other_team_session(
        self, test_app, as_admin, as_developer
    ):
        admin_client = _with_role(test_app, Role.admin, "team-admin")
        create = admin_client.post(
            "/api/v1/guardrails/sessions",
            json={"session_id": "cross-team-del", "team_id": "team-beta"},
        )
        assert create.status_code == 201, create.text

        test_app.dependency_overrides[get_current_user] = lambda: _payload(
            Role.developer, "team-alpha"
        )
        try:
            resp = as_developer.delete("/api/v1/guardrails/sessions/cross-team-del")
            assert resp.status_code == 404, (
                f"Cross-team delete allowed: {resp.status_code}"
            )
            test_app.dependency_overrides[get_current_user] = lambda: _payload(
                Role.admin, "team-admin"
            )
            still = admin_client.get("/api/v1/guardrails/sessions/cross-team-del")
            assert still.status_code == 200, "Session destroyed despite 404 reply"
        finally:
            admin_client.delete("/api/v1/guardrails/sessions/cross-team-del")


# ── Fix 4d: create_budget must not accept other-team scope_id ────────────────

class TestCreateBudgetTeamScope:
    def test_team_manager_cannot_create_budget_for_other_team(self, test_app):
        tm_client = _with_role(test_app, Role.team_manager, "team-alpha")
        try:
            resp = tm_client.post(
                "/api/v1/budgets",
                json={
                    "policy_id": "other-team-hijack",
                    "scope": "team",
                    "scope_id": "team-beta",
                    "period": "monthly",
                    "limit_usd": 1_000_000,
                    "hard_stop": False,
                },
            )
            assert resp.status_code == 403, resp.text
        finally:
            test_app.dependency_overrides.clear()

    def test_admin_can_create_budget_for_any_team(self, as_admin):
        resp = as_admin.post(
            "/api/v1/budgets",
            json={
                "policy_id": f"admin-budget-{id(self)}",
                "scope": "team",
                "scope_id": "team-beta",
                "period": "monthly",
                "limit_usd": 50.0,
                "hard_stop": True,
            },
        )
        assert resp.status_code == 201, resp.text
