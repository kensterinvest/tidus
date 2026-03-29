"""OIDC/JWKS JWT validator with a 1-hour key cache.

Supports RS256 and ES256 tokens issued by any OIDC-compliant provider
(Okta, Azure Entra ID, Google Workspace, Auth0, generic OIDC).

Typical flow:
    1. On first request, discover the JWKS URI from
       ``{issuer_url}/.well-known/openid-configuration``.
    2. Fetch and cache all public keys from the JWKS endpoint (1h TTL).
    3. Decode and validate the JWT: signature, expiry, issuer, audience.
    4. Return the raw claims dict to the caller.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import jwt
import structlog
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

log = structlog.get_logger(__name__)

_JWKS_TTL = 3600  # seconds


class JWKSCache:
    """Async JWKS fetcher with a 1-hour in-memory key cache.

    Thread-safe via an asyncio Lock — only one coroutine refreshes the
    cache at a time; all others wait for the same refresh to complete.
    """

    def __init__(self, jwks_uri: str) -> None:
        self._jwks_uri = jwks_uri
        # kid → (public_key_object, algorithm_str)
        self._keys: dict[str, tuple[Any, str]] = {}
        self._fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def is_stale(self) -> bool:
        return time.monotonic() - self._fetched_at > _JWKS_TTL

    async def _refresh(self) -> None:
        log.info("jwks_refresh", uri=self._jwks_uri)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(self._jwks_uri)
            resp.raise_for_status()
            data: dict = resp.json()

        keys: dict[str, tuple[Any, str]] = {}
        for jwk in data.get("keys", []):
            kid: str = jwk.get("kid", "__default__")
            kty: str = jwk.get("kty", "")
            alg: str = jwk.get("alg") or ("RS256" if kty == "RSA" else "ES256")

            try:
                if kty == "RSA":
                    public_key = RSAAlgorithm.from_jwk(json.dumps(jwk))
                elif kty == "EC":
                    public_key = ECAlgorithm.from_jwk(json.dumps(jwk))
                else:
                    log.warning("jwks_unsupported_kty", kid=kid, kty=kty)
                    continue
            except Exception as exc:
                log.warning("jwks_key_parse_error", kid=kid, error=str(exc))
                continue

            keys[kid] = (public_key, alg)

        self._keys = keys
        self._fetched_at = time.monotonic()
        log.info("jwks_refreshed", key_count=len(keys))

    async def get_key(self, kid: str | None) -> tuple[Any, str]:
        """Return (public_key, algorithm) for the given key ID.

        Refreshes the cache if stale or empty.  Falls back to the first
        available key when ``kid`` is None or not found after a refresh.
        """
        async with self._lock:
            if self.is_stale or not self._keys:
                await self._refresh()

        # Try exact kid match first; retry once after a forced refresh if missing
        if kid and kid not in self._keys:
            async with self._lock:
                await self._refresh()

        if kid and kid in self._keys:
            return self._keys[kid]
        if self._keys:
            return next(iter(self._keys.values()))
        raise OIDCError("No signing keys available in JWKS endpoint")


class OIDCError(Exception):
    """Raised when JWT validation fails for any reason."""


class OIDCValidator:
    """Validates OIDC JWTs using a cached JWKS.

    Args:
        issuer_url: Base URL of the OIDC provider (e.g. ``https://my.okta.com/oauth2/default``).
        client_id:  Expected ``aud`` claim value.
        team_claim: JWT claim name that holds the team identifier.
        role_claim: JWT claim name that holds the user role.

    Usage::

        validator = OIDCValidator(
            issuer_url="https://my.okta.com/oauth2/default",
            client_id="0oab...",
            team_claim="tid",
            role_claim="role",
        )
        claims = await validator.validate(token_str)
    """

    def __init__(
        self,
        issuer_url: str,
        client_id: str,
        team_claim: str = "tid",
        role_claim: str = "role",
    ) -> None:
        self._issuer_url = issuer_url.rstrip("/")
        self._client_id = client_id
        self._team_claim = team_claim
        self._role_claim = role_claim
        self._jwks_cache: JWKSCache | None = None
        self._discovery_lock = asyncio.Lock()

    async def _ensure_cache(self) -> JWKSCache:
        if self._jwks_cache is not None:
            return self._jwks_cache
        async with self._discovery_lock:
            if self._jwks_cache is not None:
                return self._jwks_cache
            jwks_uri = await self._discover_jwks_uri()
            self._jwks_cache = JWKSCache(jwks_uri)
        return self._jwks_cache

    async def _discover_jwks_uri(self) -> str:
        """Fetch the OIDC discovery document and return the jwks_uri."""
        discovery_url = f"{self._issuer_url}/.well-known/openid-configuration"
        log.info("oidc_discovery", url=discovery_url)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(discovery_url)
            resp.raise_for_status()
            doc: dict = resp.json()
        jwks_uri: str | None = doc.get("jwks_uri")
        if not jwks_uri:
            raise OIDCError(f"Discovery document at {discovery_url} missing 'jwks_uri'")
        return jwks_uri

    async def validate(self, token: str) -> dict[str, Any]:
        """Decode and validate a JWT, returning the verified claims dict.

        Raises:
            OIDCError: Token is invalid, expired, wrong issuer/audience, or
                       the JWKS endpoint is unreachable.
        """
        # Peek at the header to find the key id without verifying yet
        try:
            header = jwt.get_unverified_header(token)
        except jwt.DecodeError as exc:
            raise OIDCError(f"Malformed JWT header: {exc}") from exc

        kid: str | None = header.get("kid")

        cache = await self._ensure_cache()
        try:
            public_key, algorithm = await cache.get_key(kid)
        except OIDCError:
            raise
        except Exception as exc:
            raise OIDCError(f"JWKS key retrieval failed: {exc}") from exc

        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                key=public_key,
                algorithms=[algorithm],
                audience=self._client_id,
                issuer=self._issuer_url,
                options={"require": ["exp", "iat", "iss", "sub"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise OIDCError("Token has expired") from exc
        except jwt.InvalidAudienceError as exc:
            raise OIDCError("Token audience mismatch") from exc
        except jwt.InvalidIssuerError as exc:
            raise OIDCError("Token issuer mismatch") from exc
        except jwt.PyJWTError as exc:
            raise OIDCError(f"Token validation failed: {exc}") from exc

        return claims

    def extract_team_id(self, claims: dict[str, Any]) -> str:
        """Return the team_id from validated claims."""
        value = claims.get(self._team_claim, "")
        return str(value) if value else ""

    def extract_role(self, claims: dict[str, Any]) -> str:
        """Return the role string from validated claims."""
        value = claims.get(self._role_claim, "")
        # Some IdPs put roles in a list; take the first entry
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value) if value else ""
