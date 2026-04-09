"""add billing_reconciliations table

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-04-08

Adds the billing_reconciliations table for Phase 5 billing reconciliation.
One row per (model_id, reconciliation_date) per upload cycle.

downgrade() drops the table.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f2a3b4c5d6e7"
down_revision: str | Sequence[str] | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "billing_reconciliations",
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("reconciliation_date", sa.Date(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("uploaded_by", sa.String(), nullable=False),
        sa.Column("team_id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("tidus_cost_usd", sa.Float(), nullable=False),
        sa.Column("provider_cost_usd", sa.Float(), nullable=False),
        sa.Column("variance_usd", sa.Float(), nullable=False),
        sa.Column("variance_pct", sa.Float(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_billing_reconciliations_model_date",
        "billing_reconciliations",
        ["model_id", "reconciliation_date"],
    )
    op.create_index(
        "ix_billing_reconciliations_team_id",
        "billing_reconciliations",
        ["team_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_billing_reconciliations_team_id", table_name="billing_reconciliations")
    op.drop_index("ix_billing_reconciliations_model_date", table_name="billing_reconciliations")
    op.drop_table("billing_reconciliations")
