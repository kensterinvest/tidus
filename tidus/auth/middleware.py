"""FastAPI authentication dependency — extract and validate the caller identity.

Provides:
    TokenPayload   — Pydantic model for the authenticated caller's claims.
    get_current_user() — FastAPI dependency; validates the Bearer token or
                         falls back to dev-mode static identity when OIDC is
                         not configured.

Dev fallback:
    When ``OIDC_ISSUER_URL`` is unset (or empty), Tidus operates in dev mode:
    every request is treated as an ``admin`` caller with the team_id from
    ``OIDC_DEV_TEAM_ID`` (default: ``"team-dev"``). This preserves backward
    compatibility with pre-SSO deployments.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tidus.auth.oidc import OIDCError, OIDCValidator
from tidus.settings import Settings, get_settings

log = structlog.get_logger(__name__)

# Optional — HTTPBearer does not auto-error so we can handle missing tokens ourselves
_bearer = HTTPBearer(auto_error=False)


# ── Validated token payload ───────────────────────────────────────────────────

class TokenPayload:
    """Lightweight container for the authenticated caller's identity.

    Attributes:
        sub:         Subject identifier (user or service account ID).
        team_id:     Team this caller belongs to (drives budget enforcement).
        role:        Role string — one of the values in :class:`~tidus.auth.rbac.Role`.
        permissions: Extra permissions list extracted from the token (optional).
        raw_claims:  Full decoded JWT claims dict (for advanced use).
        tenant_id:   Tenant identifier for Stage B telemetry (per-tenant fine-
                     tuning hook). Resolution order: JWT claim → X-Tenant-ID
                     header → team_id. None when none of those resolve (dev
                     paths where team_id is also absent).
    """

    __slots__ = ("sub", "team_id", "role", "permissions", "raw_claims", "tenant_id")

    def __init__(
        self,
        sub: str,
        team_id: str,
        role: str,
        permissions: list[str],
        raw_claims: dict[str, Any],
        tenant_id: str | None = None,
    ) -> None:
        self.sub = sub
        self.team_id = team_id
        self.role = role
        self.permissions = permissions
        self.raw_claims = raw_claims
        # Default fallback: tenant IS the team for single-tier deployments.
        self.tenant_id = tenant_id or team_id or None


# ── Singleton validator (lazy-initialised) ────────────────────────────────────

@lru_cache(maxsize=1)
def _get_validator(issuer_url: str, client_id: str, team_claim: str, role_claim: str) -> OIDCValidator:
    return OIDCValidator(
        issuer_url=issuer_url,
        client_id=client_id,
        team_claim=team_claim,
        role_claim=role_claim,
    )


# ── Core dependency ───────────────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenPayload:
    """Resolve the authenticated caller from the incoming request.

    - **OIDC mode** (``OIDC_ISSUER_URL`` is set): validates the Bearer JWT
      and extracts ``team_id`` + ``role`` from the configured claims.
    - **Dev mode** (no ``OIDC_ISSUER_URL``): returns a static admin payload
      so existing integrations continue to work without any token.

    Raises:
        HTTP 401: Missing or invalid Bearer token when OIDC is configured.
        HTTP 403: Token is valid but the team or role claim is absent.
    """
    # ── Dev / no-OIDC fallback ────────────────────────────────────────────────
    if not settings.oidc_issuer_url:
        if settings.environment == "production":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "OIDC_ISSUER_URL is not configured but ENVIRONMENT=production. "
                    "Set OIDC_ISSUER_URL or switch to ENVIRONMENT=development."
                ),
            )
        log.warning(
            "auth_dev_mode_active",
            team_id=settings.oidc_dev_team_id,
            role=settings.oidc_dev_role,
            warning="All requests granted admin access — do not use in production",
        )
        # Dev mode: tenant header still allowed so local testing can exercise
        # multi-tenant log paths without OIDC.
        dev_tenant = request.headers.get(settings.tenant_header_name) or settings.oidc_dev_team_id
        return TokenPayload(
            sub="dev",
            team_id=settings.oidc_dev_team_id,
            role=settings.oidc_dev_role,
            permissions=[],
            raw_claims={},
            tenant_id=dev_tenant,
        )

    # ── OIDC mode: require Bearer token ──────────────────────────────────────
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    validator = _get_validator(
        issuer_url=settings.oidc_issuer_url,
        client_id=settings.oidc_client_id,
        team_claim=settings.oidc_team_claim,
        role_claim=settings.oidc_role_claim,
    )

    try:
        claims = await validator.validate(credentials.credentials)
    except OIDCError as exc:
        log.warning("oidc_validation_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    team_id = validator.extract_team_id(claims)
    role = validator.extract_role(claims)
    sub: str = claims.get("sub", "")

    if not team_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"JWT is missing the team claim '{settings.oidc_team_claim}'. "
                "Ensure your IdP includes the team identifier in the token."
            ),
        )
    if not role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"JWT is missing the role claim '{settings.oidc_role_claim}'. "
                "Ensure your IdP includes the Tidus role in the token."
            ),
        )

    permissions: list[str] = claims.get("permissions", []) or []

    # Tenant resolution in OIDC mode: JWT claim ONLY. We deliberately do NOT
    # consult the X-Tenant-ID header here — an authenticated user from team-A
    # could otherwise set `X-Tenant-ID: tenant-B` and poison tenant-B's
    # telemetry stream (Stage C per-tenant fine-tuning treats tenant_id as a
    # trust boundary per plan.md:554). The header fallback only exists for
    # dev mode (above) where there's no authentication to trust. If a
    # deployment needs header-based tenant propagation in prod, add tenant
    # issuance to the IdP instead.
    tenant_id = claims.get(settings.oidc_tenant_claim)

    log.debug("auth_ok", sub=sub, team_id=team_id, role=role, tenant_id=tenant_id)
    return TokenPayload(
        sub=sub,
        team_id=team_id,
        role=role,
        permissions=permissions,
        raw_claims=claims,
        tenant_id=tenant_id,
    )
