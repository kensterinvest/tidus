"""add v1.1.0 registry tables

Revision ID: d5e6f7a8b9c0
Revises: a2c4e6f8b1d3
Create Date: 2026-04-06

Adds five tables for the self-healing model registry:
  - model_catalog_revisions
  - model_catalog_entries
  - model_overrides
  - model_telemetry
  - model_drift_events

downgrade() drops them in reverse dependency order (drift→telemetry→overrides→entries→revisions).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d5e6f7a8b9c0"
down_revision: str | Sequence[str] | None = "a2c4e6f8b1d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_catalog_revisions",
        sa.Column("revision_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("activated_at", sa.DateTime(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("signature_hash", sa.String(), nullable=False, server_default=""),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("canary_results", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("revision_id"),
    )
    op.create_index("ix_model_catalog_revisions_status", "model_catalog_revisions", ["status"])

    op.create_table(
        "model_catalog_entries",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("revision_id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("spec_json", sa.JSON(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(["revision_id"], ["model_catalog_revisions.revision_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("revision_id", "model_id", name="uq_catalog_entry_rev_model"),
    )
    op.create_index("ix_model_catalog_entries_revision_id", "model_catalog_entries", ["revision_id"])

    op.create_table(
        "model_overrides",
        sa.Column("override_id", sa.String(), nullable=False),
        sa.Column("override_type", sa.String(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False, server_default="global"),
        sa.Column("scope_id", sa.String(), nullable=True),
        sa.Column("model_id", sa.String(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("owner_team_id", sa.String(), nullable=False),
        sa.Column("justification", sa.Text(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("deactivated_at", sa.DateTime(), nullable=True),
        sa.Column("deactivated_by", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("override_id"),
    )
    op.create_index("ix_model_overrides_model_active", "model_overrides", ["model_id", "is_active"])
    op.create_index("ix_model_overrides_scope_active", "model_overrides", ["scope_id", "is_active"])

    op.create_table(
        "model_telemetry",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("measured_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("latency_p50_ms", sa.Integer(), nullable=True),
        sa.Column("is_healthy", sa.Boolean(), nullable=False),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("context_exceeded_rate", sa.Float(), nullable=True),
        sa.Column("token_delta_pct", sa.Float(), nullable=True),
        sa.Column("source", sa.String(), nullable=False, server_default="health_probe"),
        sa.Column("probe_type", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_telemetry_model_measured", "model_telemetry", ["model_id", "measured_at"])

    op.create_table(
        "model_drift_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("drift_type", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("detected_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("threshold_value", sa.Float(), nullable=False),
        sa.Column("drift_status", sa.String(), nullable=False, server_default="open"),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("audit_record_id", sa.String(), nullable=True),
        sa.Column("active_revision_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["audit_record_id"], ["audit_logs.id"]),
        sa.ForeignKeyConstraint(["active_revision_id"], ["model_catalog_revisions.revision_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_drift_events_model_status", "model_drift_events", ["model_id", "drift_status"])


def downgrade() -> None:
    # Drop in reverse dependency order
    op.drop_index("ix_model_drift_events_model_status", table_name="model_drift_events")
    op.drop_table("model_drift_events")

    op.drop_index("ix_model_telemetry_model_measured", table_name="model_telemetry")
    op.drop_table("model_telemetry")

    op.drop_index("ix_model_overrides_scope_active", table_name="model_overrides")
    op.drop_index("ix_model_overrides_model_active", table_name="model_overrides")
    op.drop_table("model_overrides")

    op.drop_index("ix_model_catalog_entries_revision_id", table_name="model_catalog_entries")
    op.drop_table("model_catalog_entries")

    op.drop_index("ix_model_catalog_revisions_status", table_name="model_catalog_revisions")
    op.drop_table("model_catalog_revisions")
