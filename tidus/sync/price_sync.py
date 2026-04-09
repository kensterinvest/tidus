"""Price sync — delegates to RegistryPipeline for multi-source consensus pricing.

This module keeps the same public function signature (run_price_sync) for backward
compatibility with the scheduler and sync API endpoint. Internally it assembles the
available PricingSource list and calls RegistryPipeline.run_price_sync_cycle().

Prices verified: 2026-04-05
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


async def run_price_sync(
    registry,
    policies_path: str = "config/policies.yaml",
    session_factory=None,
) -> list[dict]:
    """Run a price sync cycle and return the list of detected changes.

    Backwards-compatible wrapper around RegistryPipeline. Returns the same
    list[dict] format as the v1.0.0 implementation for callers that only
    inspect the changes list.

    For richer results (revision_id, sources_used, ingestion_run_ids) call
    RegistryPipeline.run_price_sync_cycle() directly.
    """
    from tidus.registry.pipeline import RegistryPipeline, PipelineResult
    from tidus.settings import get_settings
    from tidus.sync.pricing.hardcoded_source import HardcodedSource

    settings = get_settings()
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
    result = await pipeline.run_price_sync_cycle(sources, policies_path=policies_path)

    if isinstance(result, PipelineResult):
        return result.changes
    return []
