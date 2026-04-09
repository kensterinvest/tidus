"""Thin async SQLAlchemy repository for registry reads and writes.

All DB access for the registry layer goes through this module so that:
1. SQL logic is isolated from business logic.
2. Tests can swap the session factory without touching higher-level classes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select, text, update

from tidus.db.registry_orm import (
    ModelCatalogEntryORM,
    ModelCatalogRevisionORM,
    ModelDriftEventORM,
    ModelOverrideORM,
)


async def get_active_revision(session_factory) -> ModelCatalogRevisionORM | None:
    async with session_factory() as session:
        result = await session.execute(
            select(ModelCatalogRevisionORM)
            .where(ModelCatalogRevisionORM.status == "active")
            .limit(1)
        )
        return result.scalars().first()


async def get_entries_for_revision(
    session_factory, revision_id: str
) -> list[ModelCatalogEntryORM]:
    async with session_factory() as session:
        result = await session.execute(
            select(ModelCatalogEntryORM).where(
                ModelCatalogEntryORM.revision_id == revision_id
            )
        )
        return result.scalars().all()


async def get_active_overrides(session_factory) -> list[ModelOverrideORM]:
    async with session_factory() as session:
        result = await session.execute(
            select(ModelOverrideORM).where(ModelOverrideORM.is_active == True)  # noqa: E712
        )
        return result.scalars().all()


async def get_override_checkpoint(session_factory) -> str:
    """Return a string that changes whenever any override is created or deactivated.

    Used by EffectiveRegistry to detect when the override layer needs rebuilding
    without fetching all rows on every 60-second refresh poll.
    """
    async with session_factory() as session:
        result = await session.execute(
            select(
                func.count(ModelOverrideORM.override_id).label("cnt"),
                func.max(ModelOverrideORM.created_at).label("max_created"),
            ).where(ModelOverrideORM.is_active == True)  # noqa: E712
        )
        row = result.one()
        cnt, max_created = row.cnt, row.max_created
        return f"{cnt}:{max_created}"


async def get_revision_by_id(
    session_factory, revision_id: str
) -> ModelCatalogRevisionORM | None:
    async with session_factory() as session:
        result = await session.execute(
            select(ModelCatalogRevisionORM).where(
                ModelCatalogRevisionORM.revision_id == revision_id
            )
        )
        return result.scalars().first()


async def get_all_revisions(session_factory) -> list[ModelCatalogRevisionORM]:
    async with session_factory() as session:
        result = await session.execute(
            select(ModelCatalogRevisionORM).order_by(
                ModelCatalogRevisionORM.created_at.desc()
            )
        )
        return result.scalars().all()


async def count_entries_for_revision(session_factory, revision_id: str) -> int:
    async with session_factory() as session:
        result = await session.execute(
            select(func.count(ModelCatalogEntryORM.id)).where(
                ModelCatalogEntryORM.revision_id == revision_id
            )
        )
        return result.scalar() or 0


async def count_entries_all_revisions(session_factory) -> dict[str, int]:
    """Return a {revision_id: entry_count} mapping in a single query.

    Replaces the N+1 pattern in list_revisions where one COUNT query was
    issued per revision.
    """
    async with session_factory() as session:
        result = await session.execute(
            select(
                ModelCatalogEntryORM.revision_id,
                func.count(ModelCatalogEntryORM.id).label("cnt"),
            ).group_by(ModelCatalogEntryORM.revision_id)
        )
        return {row.revision_id: row.cnt for row in result.all()}


async def get_open_drift_events(session_factory) -> list[ModelDriftEventORM]:
    async with session_factory() as session:
        result = await session.execute(
            select(ModelDriftEventORM).where(ModelDriftEventORM.drift_status == "open")
        )
        return result.scalars().all()


async def get_drift_event_by_id(
    session_factory, event_id: str
) -> ModelDriftEventORM | None:
    async with session_factory() as session:
        result = await session.execute(
            select(ModelDriftEventORM).where(ModelDriftEventORM.id == event_id)
        )
        return result.scalars().first()


async def deactivate_expired_overrides(session_factory) -> list[ModelOverrideORM]:
    """Batch-deactivate all overrides whose expires_at has passed.

    Returns the list of overrides that were deactivated so callers can write
    audit log entries.
    """
    now = datetime.now(UTC)
    async with session_factory() as session:
        # Fetch first so we can return the deactivated rows
        result = await session.execute(
            select(ModelOverrideORM).where(
                ModelOverrideORM.is_active == True,  # noqa: E712
                ModelOverrideORM.expires_at.is_not(None),
                ModelOverrideORM.expires_at < now,
            )
        )
        expired = result.scalars().all()

        if expired:
            ids = [o.override_id for o in expired]
            await session.execute(
                update(ModelOverrideORM)
                .where(ModelOverrideORM.override_id.in_(ids))
                .values(
                    is_active=False,
                    deactivated_at=now,
                    deactivated_by="system_expiry",
                )
            )
            await session.commit()

        return expired
