"""Registry seeder — idempotent import from models.yaml into the DB catalog.

On first startup (fresh install or fresh DB), reads config/models.yaml and
creates:
  - One `model_catalog_revisions` row with revision_id='seed-v0' (status='active', source='yaml_seed')
  - One `model_catalog_entries` row per model spec in the YAML

Idempotent: the revision uses a deterministic ID ('seed-v0'). A concurrent
second insert from another replica hits the PK unique constraint and is caught
and treated as a benign "already seeded" outcome — no application-level race
condition, the DB enforces the invariant.

Called from main.py lifespan, after create_tables() and build_singletons().

Usage:
    seeder = RegistrySeeder()
    seeded = await seeder.seed_from_yaml(get_session_factory(), "config/models.yaml")
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from tidus.db.registry_orm import ModelCatalogEntryORM, ModelCatalogRevisionORM
from tidus.router.registry import ModelRegistry

log = structlog.get_logger(__name__)

# Stable seed revision ID. Using a deterministic value means a second replica
# racing to insert hits the PK constraint and is cleanly caught rather than
# silently creating a second active revision with a different UUID.
_SEED_REVISION_ID = "seed-v0"


class RegistrySeeder:
    """Seed the model catalog DB from models.yaml on first startup."""

    async def seed_from_yaml(
        self,
        session_factory,
        models_config_path: str = "config/models.yaml",
    ) -> bool:
        """Seed the DB from YAML if no active revision exists.

        Returns True if seeding occurred, False if already seeded (idempotent).

        Thread/process safety: uses a deterministic revision_id ('seed-v0') so
        that a PK unique constraint violation from a concurrent replica insert is
        caught and treated as a benign race — both replicas end up with the same
        active revision rather than two competing ones.

        Raises:
            FileNotFoundError: if models_config_path does not exist.
            ValueError / ValidationError: if the YAML contains invalid ModelSpec data.
            Both are logged with structlog before re-raising so startup failures
            produce structured output rather than a raw traceback.
        """
        # Fast path: skip expensive file I/O if already seeded
        async with session_factory() as session:
            result = await session.execute(
                select(ModelCatalogRevisionORM)
                .where(ModelCatalogRevisionORM.revision_id == _SEED_REVISION_ID)
                .limit(1)
            )
            if result.scalars().first() is not None:
                log.info("registry_seed_skipped", reason="seed_revision_exists")
                return False

        # Load and validate YAML (CPU-bound; errors here are startup-fatal)
        try:
            registry = ModelRegistry.load(models_config_path)
        except FileNotFoundError:
            log.error("registry_seed_failed", reason="models_yaml_not_found", path=models_config_path)
            raise
        except Exception as exc:
            log.error("registry_seed_failed", reason="models_yaml_invalid", error=str(exc), path=models_config_path)
            raise

        specs = registry.list_all()

        # Serialize once; reuse for both the hash and the DB entries to guarantee
        # that spec_json stored in the DB is byte-for-byte what was hashed.
        now = datetime.now(UTC)
        spec_dicts: list[dict] = []
        for spec in specs:
            d = spec.model_dump(mode="json")
            d["schema_version"] = 1
            spec_dicts.append(d)

        signature_hash = hashlib.sha256(
            json.dumps(spec_dicts, sort_keys=True, default=str).encode()
        ).hexdigest()

        async with session_factory() as session:
            try:
                revision = ModelCatalogRevisionORM(
                    revision_id=_SEED_REVISION_ID,
                    activated_at=now,
                    source="yaml_seed",
                    signature_hash=signature_hash,
                    status="active",
                )
                session.add(revision)

                for spec, d in zip(specs, spec_dicts):
                    entry = ModelCatalogEntryORM(
                        id=str(uuid.uuid4()),
                        revision_id=_SEED_REVISION_ID,
                        model_id=spec.model_id,
                        spec_json=d,
                        schema_version=1,
                    )
                    session.add(entry)

                await session.commit()

            except IntegrityError:
                # Another replica raced us to the insert; this is not an error.
                await session.rollback()
                log.info("registry_seed_skipped", reason="concurrent_insert_detected")
                return False

        log.info(
            "registry_seeded",
            revision_id=_SEED_REVISION_ID,
            model_count=len(specs),
            source="yaml_seed",
        )
        return True
