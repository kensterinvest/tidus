"""Integration tests for the Alembic migration chain.

Tests that:
  - alembic upgrade head on a fresh DB creates all expected tables
  - alembic upgrade head on an existing v1.0.0 DB (catch-up) is a no-op
  - downgrade of registry tables removes only the 5 new tables
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config


@pytest.fixture(autouse=True)
def _isolate_database_url(monkeypatch):
    """Prevent DATABASE_URL env var from overriding the test-specific DB path.

    alembic/env.py rewrites sqlalchemy.url when DATABASE_URL is set, which
    would cause all tests to migrate the wrong database and silently pass.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)

EXPECTED_TABLES = {
    "alembic_version",
    "audit_logs",
    "cost_records",
    "budget_policies",
    "price_change_log",
    "routing_decisions",
    "ai_user_events",
    "model_catalog_revisions",
    "model_catalog_entries",
    "model_overrides",
    "model_telemetry",
    "model_drift_events",
}

REGISTRY_TABLES = {
    "model_catalog_revisions",
    "model_catalog_entries",
    "model_overrides",
    "model_telemetry",
    "model_drift_events",
}

PRE_EXISTING_TABLES = {
    "cost_records",
    "budget_policies",
    "price_change_log",
    "routing_decisions",
    "ai_user_events",
}


def _make_alembic_cfg(db_path: str) -> Config:
    """Build an Alembic Config pointing at a test SQLite DB.

    Must use sqlite+aiosqlite:// because env.py runs migrations through
    async_engine_from_config which requires an async driver.
    """
    project_root = Path(__file__).parent.parent.parent
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_path}")
    cfg.set_main_option("script_location", str(project_root / "alembic"))
    return cfg


def _get_tables(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    conn.close()
    return {r[0] for r in rows}


@pytest.fixture
def fresh_db(tmp_path):
    """Temporary SQLite DB file — completely empty."""
    return str(tmp_path / "test_tidus.db")


@pytest.fixture
def v100_db(tmp_path):
    """SQLite DB that looks like a v1.0.0 production DB.

    Has the 6 tables that create_tables() produces at startup, plus
    alembic_version pinned to the baseline revision. This simulates
    upgrading a live deployment.
    """
    db_path = str(tmp_path / "test_v100.db")
    conn = sqlite3.connect(db_path)

    # Create pre-existing tables as they existed before Alembic awareness
    conn.executescript("""
        CREATE TABLE audit_logs (
            id TEXT PRIMARY KEY,
            timestamp DATETIME,
            actor_team_id TEXT NOT NULL,
            actor_role TEXT NOT NULL,
            actor_sub TEXT NOT NULL,
            action TEXT NOT NULL,
            resource_type TEXT,
            resource_id TEXT,
            outcome TEXT NOT NULL,
            rejection_reason TEXT,
            ip_address TEXT,
            metadata JSON
        );
        CREATE TABLE cost_records (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            team_id TEXT NOT NULL,
            workflow_id TEXT,
            agent_session_id TEXT,
            agent_depth INTEGER,
            routing_decision_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            vendor TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cost_usd REAL NOT NULL,
            latency_ms REAL NOT NULL,
            timestamp DATETIME,
            fallback_used BOOLEAN,
            fallback_from TEXT
        );
        CREATE TABLE budget_policies (
            policy_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            period TEXT NOT NULL,
            limit_usd REAL NOT NULL,
            warn_at_pct REAL,
            hard_stop BOOLEAN
        );
        CREATE TABLE price_change_log (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            vendor TEXT NOT NULL,
            field_changed TEXT NOT NULL,
            old_value REAL NOT NULL,
            new_value REAL NOT NULL,
            delta_pct REAL NOT NULL,
            detected_at DATETIME,
            source TEXT
        );
        CREATE TABLE routing_decisions (
            decision_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            team_id TEXT NOT NULL,
            selected_model_id TEXT,
            selected_vendor TEXT,
            rejection_reason TEXT,
            explanation TEXT NOT NULL,
            estimated_cost_usd REAL,
            fallback_from TEXT,
            timestamp DATETIME
        );
        CREATE TABLE ai_user_events (
            id TEXT PRIMARY KEY,
            caller_id TEXT NOT NULL,
            caller_source TEXT NOT NULL,
            team_id TEXT,
            path TEXT,
            timestamp DATETIME
        );
        CREATE TABLE alembic_version (
            version_num TEXT NOT NULL,
            PRIMARY KEY (version_num)
        );
        INSERT INTO alembic_version VALUES ('f7ee6ab5176b');
    """)
    conn.commit()
    conn.close()
    return db_path


def test_fresh_db_gets_all_tables(fresh_db):
    """alembic upgrade head on an empty DB creates all 12 tables."""
    cfg = _make_alembic_cfg(fresh_db)
    command.upgrade(cfg, "head")
    tables = _get_tables(fresh_db)
    assert EXPECTED_TABLES <= tables, f"Missing: {EXPECTED_TABLES - tables}"


def test_v100_db_catchup_is_no_op_for_existing(v100_db):
    """alembic upgrade head on a v1.0.0 DB succeeds without error.

    The catch-up migration skips tables that already exist. The registry
    tables are new and should be created.
    """
    cfg = _make_alembic_cfg(v100_db)
    command.upgrade(cfg, "head")  # must not raise

    tables = _get_tables(v100_db)
    # All pre-existing tables still present (not re-created or duplicated)
    assert PRE_EXISTING_TABLES <= tables
    # New registry tables created
    assert REGISTRY_TABLES <= tables


def test_v100_db_data_preserved(v100_db):
    """Data in pre-existing tables survives the catch-up migration."""
    conn = sqlite3.connect(v100_db)
    conn.execute("INSERT INTO audit_logs (id, actor_team_id, actor_role, actor_sub, action, outcome) VALUES ('test-1','t','admin','sub','action','success')")
    conn.commit()
    conn.close()

    cfg = _make_alembic_cfg(v100_db)
    command.upgrade(cfg, "head")

    conn = sqlite3.connect(v100_db)
    rows = conn.execute("SELECT id FROM audit_logs").fetchall()
    conn.close()
    assert ("test-1",) in rows


def test_downgrade_removes_registry_tables(fresh_db):
    """Downgrading to the post-catch-up revision removes all Phase 1/3 registry tables.

    The Phase 3 migration added pricing_ingestion_runs (e1f2a3b4c5d6) on top of the
    Phase 1 registry tables (d5e6f7a8b9c0). Targeting a2c4e6f8b1d3 (post-catch-up,
    pre-registry) removes both migrations in one step and verifies all 6 registry
    tables are gone while the pre-existing tables survive.
    """
    cfg = _make_alembic_cfg(fresh_db)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "a2c4e6f8b1d3")

    tables = _get_tables(fresh_db)
    # All Phase 1 + Phase 3 registry tables gone
    all_registry = REGISTRY_TABLES | {"pricing_ingestion_runs"}
    assert not (all_registry & tables), f"Should be removed: {all_registry & tables}"
    # Catch-up tables still present
    assert "audit_logs" in tables
    assert "cost_records" in tables
