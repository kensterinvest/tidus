"""Admin sync endpoints — manually trigger health probes and price sync.

POST /api/v1/sync/health  — run health probe against all enabled models now
POST /api/v1/sync/prices  — run price sync (supports ?dry_run=true)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

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
    dry_run: bool = Query(False, description="If true, return what would change without writing anything"),
    session_factory=Depends(get_session_factory),
):
    """Run a price sync against all available pricing sources.

    Returns the detected price changes. With ?dry_run=true, shows what would
    change and any validation errors without creating a new revision.

    Response includes both the v1.0.0-compatible 'changes' field and new
    v1.1.0 fields: revision_id, sources_used, single_source_models, ingestion_run_ids.
    """
    settings = get_settings()

    from tidus.registry.pipeline import DryRunResult, PipelineResult, RegistryPipeline
    from tidus.sync.pricing.hardcoded_source import HardcodedSource

    sources = [HardcodedSource()]
    if settings.tidus_pricing_feed_url:
        from tidus.sync.pricing.feed_source import TidusPricingFeedSource
        sources.append(TidusPricingFeedSource(
            feed_url=settings.tidus_pricing_feed_url,
            signing_key=settings.tidus_pricing_feed_signing_key,
            failure_threshold=settings.pricing_feed_failure_threshold,
            reset_timeout_seconds=settings.pricing_feed_reset_timeout_seconds,
        ))

    pipeline = RegistryPipeline(session_factory, registry)
    result = await pipeline.run_price_sync_cycle(
        sources,
        policies_path=settings.policies_config_path,
        dry_run=dry_run,
    )

    if isinstance(result, DryRunResult):
        return {
            "dry_run": True,
            "changes_detected": len(result.would_change),
            "changes": result.would_change,
            "validation_errors": result.validation_errors,
        }

    if isinstance(result, PipelineResult):
        return {
            "dry_run": False,
            "changes_detected": len(result.changes),
            "changes": result.changes,           # v1.0.0 compat key
            "revision_id": result.revision_id,
            "sources_used": result.sources_used,
            "single_source_models": result.single_source_models,
            "ingestion_run_ids": result.ingestion_run_ids,
        }

    # No changes or pipeline returned None
    return {
        "dry_run": dry_run,
        "changes_detected": 0,
        "changes": [],
        "revision_id": None,
        "sources_used": [],
        "single_source_models": [],
        "ingestion_run_ids": [],
    }
