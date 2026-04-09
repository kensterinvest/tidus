"""add price_market_history table

Revision ID: a1b2c3d4e5f6
Revises: f2a3b4c5d6e7
Create Date: 2026-04-09

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6"
down_revision = "f2a3b4c5d6e7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "price_market_history",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("vendor", sa.String(), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("field", sa.String(), nullable=False),
        sa.Column("old_value_usd_1m", sa.Float(), nullable=False),
        sa.Column("new_value_usd_1m", sa.Float(), nullable=False),
        sa.Column("delta_pct", sa.Float(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("revision_id", sa.String(),
                  sa.ForeignKey("model_catalog_revisions.revision_id"), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
    )
    op.create_index("ix_price_market_history_model_date", "price_market_history",
                    ["model_id", "event_date"])
    op.create_index("ix_price_market_history_vendor_date", "price_market_history",
                    ["vendor", "event_date"])


def downgrade() -> None:
    op.drop_index("ix_price_market_history_vendor_date", "price_market_history")
    op.drop_index("ix_price_market_history_model_date", "price_market_history")
    op.drop_table("price_market_history")
