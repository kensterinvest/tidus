"""Alembic migration environment.

Supports both sync (offline) and async (online) migration modes.
The async path is required because Tidus uses SQLAlchemy's async engine.

Usage:
    # Autogenerate a new migration after model changes:
    uv run alembic revision --autogenerate -m "your description"

    # Apply all pending migrations:
    uv run alembic upgrade head

    # Downgrade one step:
    uv run alembic downgrade -1
"""

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Alembic Config object
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import Tidus ORM metadata for autogenerate support
from tidus.db.engine import Base  # noqa: E402

target_metadata = Base.metadata

# Override sqlalchemy.url from the DATABASE_URL env var if present
# (docker-compose / K8s inject this; alembic.ini holds the dev default)
if db_url := os.getenv("DATABASE_URL"):
    # Alembic needs a sync driver for the URL; map async drivers to sync
    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "sqlite+aiosqlite://", "sqlite://"
    )
    config.set_main_option("sqlalchemy.url", sync_url)


def _sync_url(url: str) -> str:
    """Convert an async driver URL to its sync equivalent for offline SQL gen."""
    return (
        url.replace("postgresql+asyncpg://", "postgresql://")
           .replace("sqlite+aiosqlite://", "sqlite://")
    )


def run_migrations_offline() -> None:
    """Generate SQL script without a live DB connection."""
    url = _sync_url(config.get_main_option("sqlalchemy.url"))
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using the async engine (required for asyncpg / aiosqlite)."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
