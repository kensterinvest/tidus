"""catchup: create pre-existing tables if they don't exist

Revision ID: a2c4e6f8b1d3
Revises: f7ee6ab5176b
Create Date: 2026-04-06

Background:
  The v1.0.0 baseline migration (f7ee6ab5176b) only created audit_logs.
  The other five tables were created by create_tables() at startup outside
  Alembic's awareness. This catch-up migration formalises them under Alembic
  control using IF NOT EXISTS semantics so it is a no-op on any live v1.0.0 DB.

  downgrade() is intentionally a no-op — we never drop production data tables.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a2c4e6f8b1d3"
down_revision: str | Sequence[str] | None = "f7ee6ab5176b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _existing_tables() -> set[str]:
    """Return the set of table names using a dialect-safe, greenlet-compatible approach.

    SQLAlchemy's inspect(bind).get_table_names() is known to trigger
    MissingGreenlet errors with asyncpg when called inside run_sync migration
    contexts. op.get_context().dialect.has_table() makes a direct SQL call per
    table and is explicitly supported in both sync and async-bridged contexts.
    """
    ctx = op.get_context()
    bind = op.get_bind()
    dialect = ctx.dialect
    return {
        name
        for name in (
            "cost_records",
            "budget_policies",
            "price_change_log",
            "routing_decisions",
            "ai_user_events",
        )
        if dialect.has_table(bind, name)
    }


def upgrade() -> None:
    existing_tables = _existing_tables()

    if "cost_records" not in existing_tables:
        op.create_table(
            "cost_records",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("task_id", sa.String(), nullable=False),
            sa.Column("team_id", sa.String(), nullable=False),
            sa.Column("workflow_id", sa.String(), nullable=True),
            sa.Column("agent_session_id", sa.String(), nullable=True),
            sa.Column("agent_depth", sa.Integer(), nullable=True),
            sa.Column("routing_decision_id", sa.String(), nullable=False),
            sa.Column("model_id", sa.String(), nullable=False),
            sa.Column("vendor", sa.String(), nullable=False),
            sa.Column("input_tokens", sa.Integer(), nullable=False),
            sa.Column("output_tokens", sa.Integer(), nullable=False),
            sa.Column("cost_usd", sa.Float(), nullable=False),
            sa.Column("latency_ms", sa.Float(), nullable=False),
            sa.Column("timestamp", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
            sa.Column("fallback_used", sa.Boolean(), nullable=True),
            sa.Column("fallback_from", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_cost_records_task_id", "cost_records", ["task_id"])
        op.create_index("ix_cost_records_team_id", "cost_records", ["team_id"])
        op.create_index("ix_cost_records_workflow_id", "cost_records", ["workflow_id"])
        op.create_index("ix_cost_records_model_id", "cost_records", ["model_id"])
        op.create_index("ix_cost_records_timestamp", "cost_records", ["timestamp"])

    if "budget_policies" not in existing_tables:
        op.create_table(
            "budget_policies",
            sa.Column("policy_id", sa.String(), nullable=False),
            sa.Column("scope", sa.String(), nullable=False),
            sa.Column("scope_id", sa.String(), nullable=False),
            sa.Column("period", sa.String(), nullable=False),
            sa.Column("limit_usd", sa.Float(), nullable=False),
            sa.Column("warn_at_pct", sa.Float(), nullable=True),
            sa.Column("hard_stop", sa.Boolean(), nullable=True),
            sa.PrimaryKeyConstraint("policy_id"),
        )
        op.create_index("ix_budget_policies_scope_id", "budget_policies", ["scope_id"])

    if "price_change_log" not in existing_tables:
        op.create_table(
            "price_change_log",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("model_id", sa.String(), nullable=False),
            sa.Column("vendor", sa.String(), nullable=False),
            sa.Column("field_changed", sa.String(), nullable=False),
            sa.Column("old_value", sa.Float(), nullable=False),
            sa.Column("new_value", sa.Float(), nullable=False),
            sa.Column("delta_pct", sa.Float(), nullable=False),
            sa.Column("detected_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
            sa.Column("source", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_price_change_log_model_id", "price_change_log", ["model_id"])

    if "routing_decisions" not in existing_tables:
        op.create_table(
            "routing_decisions",
            sa.Column("decision_id", sa.String(), nullable=False),
            sa.Column("task_id", sa.String(), nullable=False),
            sa.Column("team_id", sa.String(), nullable=False),
            sa.Column("selected_model_id", sa.String(), nullable=True),
            sa.Column("selected_vendor", sa.String(), nullable=True),
            sa.Column("rejection_reason", sa.String(), nullable=True),
            sa.Column("explanation", sa.Text(), nullable=False),
            sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
            sa.Column("fallback_from", sa.String(), nullable=True),
            sa.Column("timestamp", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
            sa.PrimaryKeyConstraint("decision_id"),
        )
        op.create_index("ix_routing_decisions_task_id", "routing_decisions", ["task_id"])
        op.create_index("ix_routing_decisions_team_id", "routing_decisions", ["team_id"])
        op.create_index("ix_routing_decisions_timestamp", "routing_decisions", ["timestamp"])

    if "ai_user_events" not in existing_tables:
        op.create_table(
            "ai_user_events",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("caller_id", sa.String(), nullable=False),
            sa.Column("caller_source", sa.String(), nullable=False),
            sa.Column("team_id", sa.String(), nullable=True),
            sa.Column("path", sa.String(), nullable=True),
            sa.Column("timestamp", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_ai_user_events_team_id", "ai_user_events", ["team_id"])
        op.create_index("ix_ai_user_events_caller_ts", "ai_user_events", ["caller_id", "timestamp"])
        op.create_index("ix_ai_user_events_ts", "ai_user_events", ["timestamp"])


def downgrade() -> None:
    # Intentionally a no-op: these tables hold production data; dropping them
    # would be destructive. Revert by running upgrade() on the previous migration.
    pass
