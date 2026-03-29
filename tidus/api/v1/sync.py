"""Admin sync endpoints — manually trigger health probes and price sync.

These are admin-only operations (no auth yet — auth stub in deps.py).
Useful for testing or forcing an immediate sync outside the scheduled window.

POST /api/v1/sync/health  — run health probe against all enabled models now
POST /api/v1/sync/prices  — run price sync against known-prices dict now
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from tidus.api.deps import get_registry, get_session_factory
from tidus.auth.middleware import TokenPayload
from tidus.auth.rbac import Role, require_role
from tidus.settings import get_settings

router = APIRouter(tags=["Sync (Admin)"])


@router.post(
    "/sync/health",
    summary="Trigger health probe (admin)",
    response_model=dict,
)
async def trigger_health_probe(
    registry=Depends(get_registry),
    _auth: Annotated[TokenPayload, Depends(require_role(Role.admin))] = None,
):
    """Run a health probe against all enabled models immediately.

    Returns per-model health results: {model_id: is_healthy}.
    """
    settings = get_settings()
    from tidus.sync.health_probe import HealthProbe

    probe = HealthProbe(registry, settings.policies_config_path)
    results = await probe.run_once()
    healthy = sum(1 for v in results.values() if v)
    return {
        "probed": len(results),
        "healthy": healthy,
        "unhealthy": len(results) - healthy,
        "results": results,
    }


@router.post(
    "/sync/prices",
    summary="Trigger price sync (admin)",
    response_model=dict,
)
async def trigger_price_sync(
    registry=Depends(get_registry),
    _auth: Annotated[TokenPayload, Depends(require_role(Role.admin))] = None,
):
    """Run a price sync against the known-prices reference immediately.

    Returns list of price changes detected (empty list = all prices match).
    """
    settings = get_settings()
    from tidus.sync.price_sync import run_price_sync

    session_factory = get_session_factory()
    changes = await run_price_sync(
        registry,
        policies_path=settings.policies_config_path,
        session_factory=session_factory,
    )
    return {
        "changes_detected": len(changes),
        "changes": changes,
    }
