"""Unit tests for tidus.auth.rbac — _has_role behavior.

Fix 5 regression: admin must satisfy ANY required role even when the callsite
forgets to list Role.admin explicitly. Defense-in-depth against silent admin
denial bugs.
"""

from __future__ import annotations

from tidus.auth.rbac import Role, _has_role


class TestAdminSatisfiesAny:
    """Admin is a super-role: it satisfies any required role list."""

    def test_admin_satisfies_developer_requirement(self):
        assert _has_role(Role.admin.value, Role.developer) is True

    def test_admin_satisfies_team_manager_requirement(self):
        assert _has_role(Role.admin.value, Role.team_manager) is True

    def test_admin_satisfies_read_only_requirement(self):
        assert _has_role(Role.admin.value, Role.read_only) is True

    def test_admin_satisfies_when_not_in_list(self):
        """Callsite lists only team_manager+developer, admin must still pass."""
        assert _has_role(Role.admin.value, Role.team_manager, Role.developer) is True

    def test_admin_satisfies_empty_required_list(self):
        """Defensive: even an empty required list should let admin through."""
        assert _has_role(Role.admin.value) is True


class TestNonAdminExactMatch:
    """Non-admin roles must appear in the required list."""

    def test_developer_satisfies_developer(self):
        assert _has_role(Role.developer.value, Role.developer) is True

    def test_developer_does_not_satisfy_admin_requirement(self):
        assert _has_role(Role.developer.value, Role.admin) is False

    def test_developer_does_not_satisfy_team_manager_requirement(self):
        assert _has_role(Role.developer.value, Role.team_manager) is False

    def test_read_only_does_not_satisfy_developer(self):
        assert _has_role(Role.read_only.value, Role.developer) is False

    def test_service_account_satisfies_service_account(self):
        assert _has_role(Role.service_account.value, Role.service_account) is True

    def test_team_manager_satisfies_team_manager(self):
        assert _has_role(Role.team_manager.value, Role.team_manager) is True


class TestInvalidRole:
    """Unknown role strings return False."""

    def test_unknown_role_rejected(self):
        assert _has_role("superuser", Role.admin) is False

    def test_empty_role_rejected(self):
        assert _has_role("", Role.developer) is False
