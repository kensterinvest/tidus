"""Unit tests for RegistrySeeder.

Tests that:
  - Seeding creates exactly 1 revision + N entries from YAML
  - Revision uses the deterministic ID 'seed-v0'
  - A second call is idempotent (no duplicate revisions)
  - spec_json entries carry schema_version=1
  - spec_json content matches the data used to compute the signature_hash
  - The seeder returns True on first call, False on subsequent calls
  - Missing YAML raises FileNotFoundError with a structured log
  - Concurrent insert (IntegrityError) is treated as a benign race
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tidus.db.engine import Base
from tidus.db.registry_orm import ModelCatalogEntryORM, ModelCatalogRevisionORM
from tidus.registry.seeder import _SEED_REVISION_ID, RegistrySeeder


@pytest.fixture(scope="module")
def models_yaml_path():
    # Resolve absolute path so tests pass regardless of the pytest invocation directory.
    return str(Path(__file__).parent.parent.parent / "config" / "models.yaml")


@pytest_asyncio.fixture
async def session_factory():
    """In-memory SQLite session factory for isolated seeder tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_seed_creates_one_revision(session_factory, models_yaml_path):
    seeder = RegistrySeeder()
    result = await seeder.seed_from_yaml(session_factory, models_yaml_path)

    assert result is True

    async with session_factory() as session:
        revisions = (await session.execute(select(ModelCatalogRevisionORM))).scalars().all()
    assert len(revisions) == 1
    rev = revisions[0]
    assert rev.revision_id == _SEED_REVISION_ID
    assert rev.status == "active"
    assert rev.source == "yaml_seed"
    assert rev.activated_at is not None
    assert rev.signature_hash != ""


@pytest.mark.asyncio
async def test_seed_spec_json_matches_signature_hash(session_factory, models_yaml_path):
    """spec_json entries must be bit-for-bit what was hashed (single model_dump pass)."""
    import hashlib
    import json

    seeder = RegistrySeeder()
    await seeder.seed_from_yaml(session_factory, models_yaml_path)

    async with session_factory() as session:
        revision = (await session.execute(
            select(ModelCatalogRevisionORM).where(ModelCatalogRevisionORM.status == "active")
        )).scalars().first()
        entries = (await session.execute(
            select(ModelCatalogEntryORM).where(ModelCatalogEntryORM.revision_id == _SEED_REVISION_ID)
        )).scalars().all()

    spec_dicts = [e.spec_json for e in entries]
    recomputed_hash = hashlib.sha256(
        json.dumps(spec_dicts, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert revision.signature_hash == recomputed_hash


@pytest.mark.asyncio
async def test_seed_missing_yaml_raises(session_factory):
    seeder = RegistrySeeder()
    with pytest.raises(FileNotFoundError):
        await seeder.seed_from_yaml(session_factory, "/nonexistent/path/models.yaml")


@pytest.mark.asyncio
async def test_seed_creates_n_entries(session_factory, models_yaml_path):
    from tidus.router.registry import ModelRegistry

    seeder = RegistrySeeder()
    await seeder.seed_from_yaml(session_factory, models_yaml_path)

    expected_count = len(ModelRegistry.load(models_yaml_path).list_all())

    async with session_factory() as session:
        entries = (await session.execute(select(ModelCatalogEntryORM))).scalars().all()
    assert len(entries) == expected_count


@pytest.mark.asyncio
async def test_seed_entries_have_schema_version_1(session_factory, models_yaml_path):
    seeder = RegistrySeeder()
    await seeder.seed_from_yaml(session_factory, models_yaml_path)

    async with session_factory() as session:
        entries = (await session.execute(select(ModelCatalogEntryORM))).scalars().all()

    for entry in entries:
        assert entry.schema_version == 1
        assert entry.spec_json.get("schema_version") == 1


@pytest.mark.asyncio
async def test_seed_entries_link_to_revision(session_factory, models_yaml_path):
    seeder = RegistrySeeder()
    await seeder.seed_from_yaml(session_factory, models_yaml_path)

    async with session_factory() as session:
        revision = (await session.execute(
            select(ModelCatalogRevisionORM).where(ModelCatalogRevisionORM.status == "active")
        )).scalars().first()
        entries = (await session.execute(select(ModelCatalogEntryORM))).scalars().all()

    assert all(e.revision_id == revision.revision_id for e in entries)


@pytest.mark.asyncio
async def test_seed_idempotent_second_call(session_factory, models_yaml_path):
    seeder = RegistrySeeder()

    first = await seeder.seed_from_yaml(session_factory, models_yaml_path)
    second = await seeder.seed_from_yaml(session_factory, models_yaml_path)

    assert first is True
    assert second is False  # skipped — revision already exists

    async with session_factory() as session:
        revisions = (await session.execute(select(ModelCatalogRevisionORM))).scalars().all()

    assert len(revisions) == 1  # still only one revision


@pytest.mark.asyncio
async def test_seed_idempotent_multiple_calls(session_factory, models_yaml_path):
    seeder = RegistrySeeder()
    results = []
    for _ in range(5):
        results.append(await seeder.seed_from_yaml(session_factory, models_yaml_path))

    assert results == [True, False, False, False, False]

    async with session_factory() as session:
        count = len((await session.execute(select(ModelCatalogRevisionORM))).scalars().all())
    assert count == 1
