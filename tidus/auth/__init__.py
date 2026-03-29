"""Tidus authentication and RBAC package.

Provides:
    - OIDCValidator: async JWKS-backed JWT validator
    - get_current_user: FastAPI dependency for authenticated user context
    - require_role: dependency factory for per-route RBAC enforcement
"""
