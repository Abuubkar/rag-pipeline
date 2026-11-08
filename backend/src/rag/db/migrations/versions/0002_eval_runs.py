"""Evaluation runs and per-question results.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-08
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON, JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("mode", sa.String, nullable=False),
        sa.Column("split", sa.String, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("questions", sa.Integer, nullable=False, server_default="0"),
        sa.Column("index_version_id", sa.String, sa.ForeignKey("index_versions.id"), nullable=False),
        sa.Column("config_hash", sa.String, nullable=False),
        sa.Column("config", JSONB, nullable=False),
        sa.Column("provenance", JSONB, nullable=False),
        sa.Column("summary", JSON),  # JSON keeps key order (metrics in reading order); JSONB sorts keys
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "eval_results",
        sa.Column("run_id", sa.String, sa.ForeignKey("eval_runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("item_id", sa.String, primary_key=True),
        sa.Column("question_type", sa.String, nullable=False),
        sa.Column("scores", JSONB, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("eval_results")
    op.drop_table("eval_runs")
