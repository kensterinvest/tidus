"""RBAC role definitions and the ``require_role`` dependency factory.

Roles (least → most privileged):
    read_only       — dashboard read access only
    service_account — workflow-scoped API calls (route/complete)
    developer       — route, complete, view own team usage
    team_manager    — budget management + developer permissions
    admin           — full access including model registry and sync ops
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from fastapi import Depends, HTTPException, status

from tidus.auth.middleware import TokenPayload, get_current_user


class Role(str, Enum):
    read_only = "read_only"
    service_account = "service_account"
    developer = "developer"
    team_manager = "team_manager"
    admin = "admin"


# Ordered hierarchy — each role includes all permissions of roles below it.
_ROLE_HIERARCHY: list[Role] = [
    Role.read_only,
    Role.service_account,
    Role.developer,
    Role.team_manager,
    Role.admin,
]

# Roles that may send AI traffic (route/complete endpoints)
AI_TRAFFIC_ROLES: frozenset[Role] = frozenset({
    Role.developer,
    Role.team_manager,
    Role.admin,
    Role.service_account,
})


def _has_role(actual: str, *required: Role) -> bool:
    """Return True if the actual role satisfies any of the required roles."""
    try:
        actual_role = Role(actual)
    except ValueError:
        return False
    return actual_role in required


def require_role(*roles: Role):
    """FastAPI dependency factory — enforce that the caller holds one of ``roles``.

    Usage::

        @router.post("/admin-only")
        async def handler(
            _: Annotated[TokenPayload, Depends(require_role(Role.admin))],
        ): ...

    Args:
        *roles: One or more :class:`Role` values that are permitted.

    Returns:
        A FastAPI dependency (callable) that resolves to the :class:`TokenPayload`
        on success or raises HTTP 403 on failure.
    """
    async def _check(
        user: Annotated[TokenPayload, Depends(get_current_user)],
    ) -> TokenPayload:
        if not _has_role(user.role, *roles):
            allowed = ", ".join(r.value for r in roles)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not authorised. Required: {allowed}",
            )
        return user

    return _check
