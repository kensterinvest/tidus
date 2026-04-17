"""Unit tests for RegistryPipeline.

Covers:
  - Price change above threshold → new ACTIVE revision created
  - Identical prices (no change) → returns None, revision count unchanged
  - No active revision in DB → returns None (cannot diff)
  - Tier 1/2 validation failure → returns None, no revision created
  - Tier 3 canary failure → PENDING revision marked FAILED, old ACTIVE revision preserved
  - force_activate: promotes SUPERSEDED revision, skips Tier 3
  - dry_run=True: returns DryRunResult, no DB writes
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tidus.auth.middleware import TokenPayload
from tidus.db.engine import Base
from tidus.db.registry_orm import ModelCatalogEntryORM, ModelCatalogRevisionORM
from tidus.registry.pipeline import DryRunResult, PipelineResult, RegistryPipeline
from tidus.sync.pricing.base import PriceQuote, PricingSource

_POLICIES_PATH = str(Path(__file__).parent.parent.parent / "config" / "policies.yaml")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def sf():
    """In-memory SQLite session factory with all tables created."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _spec_dict(model_id: str = "gpt-4o", input_price: float = 5.0) -> dict:
    """Minimal valid ModelSpec JSON for catalog entries."""
    return {
        "model_id": model_id,
        "display_name": model_id,
        "vendor": "openai",
        "tier": 2,
        "max_context": 128_000,
        "input_price": input_price,
        "output_price": 10.0,
        "cache_read_price": 0.0,
        "cache_write_price": 0.0,
        "tokenizer": "tiktoken_o200k",
        "is_local": False,
        "enabled": True,
        "deprecated": False,
        "capabilities": ["chat"],
        "min_complexity": "simple",
        "max_complexity": "critical",
        "fallbacks": [],
        "schema_version": 1,
    }


async def _seed_active_revision(sf, model_id: str = "gpt-4o", input_price: float = 5.0) -> str:
    """Insert a minimal active revision with one model entry. Returns revision_id."""
    revision_id = f"rev-{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    async with sf() as session:
        session.add(ModelCatalogRevisionORM(
            revision_id=revision_id,
            source="yaml_seed",
            signature_hash="abc123",
            status="active",
            activated_at=now,
        ))
        session.add(ModelCatalogEntryORM(
            id=str(uuid.uuid4()),
            revision_id=revision_id,
            model_id=model_id,
            spec_json=_spec_dict(model_id, input_price),
            schema_version=1,
        ))
        await session.commit()
    return revision_id


def _make_quote(model_id: str = "gpt-4o", input_price: float = 5.0) -> PriceQuote:
    return PriceQuote(
        model_id=model_id,
        input_price=input_price,
        output_price=10.0,
        cache_read_price=0.0,
        cache_write_price=0.0,
        currency="USD",
        effective_date=date.today(),
        retrieved_at=datetime.now(UTC),
        source_name="test_source",
        source_confidence=0.9,
    )


class _SimpleSource(PricingSource):
    """Minimal PricingSource for tests."""

    def __init__(self, quotes: list[PriceQuote]) -> None:
        self._quotes = quotes

    @property
    def source_name(self) -> str:
        return "test_source"

    @property
    def confidence(self) -> float:
        return 0.9

    async def fetch_quotes(self) -> list[PriceQuote]:
        return self._quotes


def _make_actor() -> TokenPayload:
    return TokenPayload(sub="admin@test.com", team_id="team-a", role="admin", permissions=[], raw_claims={})


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_creates_revision_on_price_change(sf):
    """A 100% price increase (above 5% threshold) creates a new ACTIVE revision."""
    await _seed_active_revision(sf, input_price=5.0)
    source = _SimpleSource([_make_quote(input_price=10.0)])  # 100% increase

    with patch("tidus.registry.validators.CanaryProbe.run", new_callable=AsyncMock) as mock_canary:
        mock_canary.return_value = (True, [])
        result = await RegistryPipeline(sf).run_price_sync_cycle([source], _POLICIES_PATH)

    assert isinstance(result, PipelineResult)
    assert result.revision_id is not None
    assert len(result.changes) >= 1
    assert any(c["model_id"] == "gpt-4o" for c in result.changes)

    # The new revision should be ACTIVE
    async with sf() as session:
        active = (await session.execute(
            select(ModelCatalogRevisionORM).where(ModelCatalogRevisionORM.status == "active")
        )).scalars().all()
    assert len(active) == 1
    assert active[0].revision_id == result.revision_id


@pytest.mark.asyncio
async def test_pipeline_returns_none_when_no_changes(sf):
    """Identical prices (delta=0) produce no new revision."""
    await _seed_active_revision(sf, input_price=5.0)
    source = _SimpleSource([_make_quote(input_price=5.0)])  # same price

    with patch("tidus.registry.validators.CanaryProbe.run", new_callable=AsyncMock):
        result = await RegistryPipeline(sf).run_price_sync_cycle([source], _POLICIES_PATH)

    assert result is None

    # Only the seed revision exists
    async with sf() as session:
        revisions = (await session.execute(select(ModelCatalogRevisionORM))).scalars().all()
    assert len(revisions) == 1


@pytest.mark.asyncio
async def test_pipeline_retires_models_removed_from_yaml(sf):
    """Fix 7 regression: a model in the DB revision that no longer exists in
    models.yaml must be dropped from the new revision and recorded as retired.
    """
    # Seed two models: gpt-4o (real, exists in YAML) and ghost-model-v1 (fake)
    revision_id = f"rev-{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    async with sf() as session:
        session.add(ModelCatalogRevisionORM(
            revision_id=revision_id, source="yaml_seed",
            signature_hash="abc", status="active", activated_at=now,
        ))
        session.add(ModelCatalogEntryORM(
            id=str(uuid.uuid4()), revision_id=revision_id,
            model_id="gpt-4o", spec_json=_spec_dict("gpt-4o", 5.0), schema_version=1,
        ))
        session.add(ModelCatalogEntryORM(
            id=str(uuid.uuid4()), revision_id=revision_id,
            model_id="ghost-model-v1",
            spec_json=_spec_dict("ghost-model-v1", 1.0), schema_version=1,
        ))
        await session.commit()

    # Pricing source returns a valid quote for gpt-4o (triggers a price delta to advance the pipeline)
    source = _SimpleSource([_make_quote("gpt-4o", input_price=10.0)])

    with patch("tidus.registry.validators.CanaryProbe.run", new_callable=AsyncMock) as mock_canary:
        mock_canary.return_value = (True, [])
        result = await RegistryPipeline(sf).run_price_sync_cycle([source], _POLICIES_PATH)

    assert result is not None, "Pipeline should create a revision (price change + retirement)"
    retired_changes = [c for c in result.changes if c.get("field") == "retired"]
    assert any(c["model_id"] == "ghost-model-v1" for c in retired_changes), (
        f"Expected 'retired' change for ghost-model-v1, got changes: {result.changes}"
    )

    # The new revision must NOT contain ghost-model-v1
    async with sf() as session:
        entries = (await session.execute(
            select(ModelCatalogEntryORM).where(
                ModelCatalogEntryORM.revision_id == result.revision_id
            )
        )).scalars().all()
    model_ids = {e.model_id for e in entries}
    assert "ghost-model-v1" not in model_ids, (
        f"Retired model survived into new revision: {model_ids}"
    )
    assert "gpt-4o" in model_ids, "gpt-4o must survive — it's still in YAML"


@pytest.mark.asyncio
async def test_pipeline_returns_none_when_no_active_revision(sf):
    """Without an active revision in DB, the pipeline cannot compute a diff."""
    source = _SimpleSource([_make_quote(input_price=5.0)])

    result = await RegistryPipeline(sf).run_price_sync_cycle([source], _POLICIES_PATH)
    assert result is None


@pytest.mark.asyncio
async def test_pipeline_returns_none_on_tier2_validation_failure(sf):
    """Tier 2 violation (local model with price > 0) → returns None, no revision created."""
    # Seed a local model that is currently free
    local_spec = {**_spec_dict("gemini-nano", input_price=0.0), "is_local": True, "output_price": 0.0}
    revision_id = f"rev-{uuid.uuid4().hex[:8]}"
    async with sf() as session:
        session.add(ModelCatalogRevisionORM(
            revision_id=revision_id, source="yaml_seed",
            signature_hash="abc", status="active", activated_at=datetime.now(UTC),
        ))
        session.add(ModelCatalogEntryORM(
            id=str(uuid.uuid4()), revision_id=revision_id,
            model_id="gemini-nano", spec_json=local_spec, schema_version=1,
        ))
        await session.commit()

    # Source returns a price for the local model — this violates InvariantValidator
    quote = PriceQuote(
        model_id="gemini-nano",
        input_price=5.0,  # local model should be free!
        output_price=10.0,
        cache_read_price=0.0, cache_write_price=0.0,
        currency="USD", effective_date=date.today(),
        retrieved_at=datetime.now(UTC),
        source_name="test_source", source_confidence=0.9,
    )
    source = _SimpleSource([quote])

    result = await RegistryPipeline(sf).run_price_sync_cycle([source], _POLICIES_PATH)
    assert result is None

    # No new revision created beyond the seed
    async with sf() as session:
        revisions = (await session.execute(select(ModelCatalogRevisionORM))).scalars().all()
    assert len(revisions) == 1


@pytest.mark.asyncio
async def test_pipeline_canary_failure_marks_revision_failed_keeps_old_active(sf):
    """Tier 3 canary failure: new revision is FAILED, old revision stays ACTIVE."""
    seed_rev_id = await _seed_active_revision(sf, input_price=5.0)
    source = _SimpleSource([_make_quote(input_price=10.0)])  # triggers a change

    with patch("tidus.registry.validators.CanaryProbe.run", new_callable=AsyncMock) as mock_canary:
        mock_canary.return_value = (False, [])  # canary fails
        result = await RegistryPipeline(sf).run_price_sync_cycle([source], _POLICIES_PATH)

    assert result is None

    async with sf() as session:
        revisions = (await session.execute(select(ModelCatalogRevisionORM))).scalars().all()

    # Should have 2 revisions: the seed (ACTIVE) and a new one (FAILED)
    assert len(revisions) == 2
    by_status = {r.status: r for r in revisions}
    assert "active" in by_status
    assert "failed" in by_status
    assert by_status["active"].revision_id == seed_rev_id


@pytest.mark.asyncio
async def test_pipeline_force_activate_promotes_superseded_revision(sf):
    """force_activate runs Tier 1 + Tier 2 only and promotes a SUPERSEDED revision."""
    await _seed_active_revision(sf, input_price=5.0)

    # Create a SUPERSEDED revision manually
    superseded_id = f"rev-{uuid.uuid4().hex[:8]}"
    async with sf() as session:
        session.add(ModelCatalogRevisionORM(
            revision_id=superseded_id, source="price_sync",
            signature_hash="xyz", status="superseded",
        ))
        session.add(ModelCatalogEntryORM(
            id=str(uuid.uuid4()), revision_id=superseded_id,
            model_id="gpt-4o", spec_json=_spec_dict("gpt-4o", 7.0), schema_version=1,
        ))
        await session.commit()

    pipeline = RegistryPipeline(sf)
    await pipeline.force_activate(superseded_id, _make_actor(), "Testing force activate")

    async with sf() as session:
        revisions = (await session.execute(select(ModelCatalogRevisionORM))).scalars().all()
    by_status = {r.status: r for r in revisions}

    # The previously superseded revision is now ACTIVE
    assert by_status["active"].revision_id == superseded_id
    # The previously active revision is now SUPERSEDED
    assert "superseded" in by_status


@pytest.mark.asyncio
async def test_pipeline_dry_run_returns_dry_run_result_no_db_writes(sf):
    """dry_run=True returns DryRunResult describing what would change without writing anything."""
    await _seed_active_revision(sf, input_price=5.0)
    source = _SimpleSource([_make_quote(input_price=10.0)])  # 100% increase

    result = await RegistryPipeline(sf).run_price_sync_cycle(
        [source], _POLICIES_PATH, dry_run=True
    )

    assert isinstance(result, DryRunResult)
    assert len(result.would_change) >= 1
    assert result.validation_errors == []

    # No revision should have been created (only the seed)
    async with sf() as session:
        revisions = (await session.execute(select(ModelCatalogRevisionORM))).scalars().all()
    assert len(revisions) == 1
