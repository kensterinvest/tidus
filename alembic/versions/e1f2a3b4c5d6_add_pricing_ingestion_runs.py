"""add pricing_ingestion_runs table

Revision ID: e1f2a3b4c5d6
Revises: d5e6f7a8b9c0
Create Date: 2026-04-06

Adds the pricing_ingestion_runs table for Phase 3 multi-source pricing.
One row per source per sync cycle — complete audit trail for price changes.

downgrade() drops the table.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: str | Sequence[str] | None = "d5e6f7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pricing_ingestion_runs",
        sa.Column("run_id", sa.String(), primary_key=True, nullable=False),
        sa.Column("started_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("source_name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="success"),
        sa.Column("raw_payload", sa.JSON(), nullable=True),
        sa.Column("model_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("quotes_valid", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("quotes_rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rejection_reasons", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("revision_id_created", sa.String(), sa.ForeignKey("model_catalog_revisions.revision_id"), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("pricing_ingestion_runs")
