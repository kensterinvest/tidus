"""OverrideExpiryJob — batch-deactivate expired overrides on a 15-minute schedule.

Runs every 15 minutes (configurable). Queries for active overrides whose
expires_at has passed, deactivates them via a single batch UPDATE, writes an
audit entry per deactivated row, then triggers EffectiveRegistry.refresh() so
the change propagates to the merge layer immediately.

This closes the gap where expires_at fields exist in the schema but nothing
enforces them (they would silently linger as phantom restrictions otherwise).
"""

from __future__ import annotations

import structlog

from tidus.audit.logger import AuditLogger
from tidus.auth.middleware import TokenPayload
from tidus.db.repositories.registry_repo import deactivate_expired_overrides

log = structlog.get_logger(__name__)

# Synthetic actor for audit log entries written by the system expiry job.
_SYSTEM_ACTOR = TokenPayload(
    sub="system_expiry",
    team_id="system",
    role="admin",
    permissions=[],
    raw_claims={},
)


class OverrideExpiryJob:
    """Deactivates expired overrides and writes audit entries."""

    async def run(self, session_factory, registry=None) -> int:
        """Deactivate all overrides whose expires_at has passed.

        Args:
            session_factory: Async SQLAlchemy session factory.
            registry:        Optional EffectiveRegistry — if provided, refresh()
                             is called after deactivation so stale overrides are
                             removed from the merge layer immediately.

        Returns:
            Number of overrides deactivated (0 when none have expired).
        """
        try:
            expired = await deactivate_expired_overrides(session_factory)
        except Exception as exc:
            log.error("override_expiry_job_failed", error=str(exc))
            return 0

        if not expired:
            return 0

        log.info("override_expiry_deactivated", count=len(expired))

        # Write one audit entry per expired override (non-fatal)
        audit = AuditLogger(session_factory)
        for override in expired:
            await audit.record(
                actor=_SYSTEM_ACTOR,
                action="registry.override_expired",
                resource_type="model_override",
                resource_id=override.override_id,
                outcome="success",
                metadata={
                    "override_type": override.override_type,
                    "model_id": override.model_id,
                    "owner_team_id": override.owner_team_id,
                },
            )

        # Propagate immediately — don't wait for the next 60-second refresh cycle
        if registry is not None:
            try:
                refreshed = await registry.refresh(session_factory)
                if refreshed:
                    log.info("override_expiry_registry_refreshed")
            except Exception as exc:
                log.error("override_expiry_refresh_failed", error=str(exc))

        return len(expired)
