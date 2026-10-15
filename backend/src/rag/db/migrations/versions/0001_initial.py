"""Initial schema: documents, index versions, chunks with pgvector + full-text search.

Revision ID: 0001
Revises:
Create Date: 2026-10-16
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("source", sa.String),
        sa.Column("category", sa.String),
        sa.Column("author", sa.String),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("url", sa.Text),
        sa.Column("content_hash", sa.String, nullable=False),
    )

    op.create_table(
        "index_versions",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("manifest", JSONB, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("expected_chunks", sa.Integer),
        sa.Column("indexed_chunks", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "active_index",
        sa.Column("alias", sa.String, primary_key=True),
        sa.Column("version_id", sa.String, sa.ForeignKey("index_versions.id"), nullable=False),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "version_id",
            sa.String,
            sa.ForeignKey("index_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("document_id", sa.String, sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column(
            "tsv",
            TSVECTOR,
            sa.Computed("to_tsvector('english', text)", persisted=True),
        ),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_chunks_version", "chunks", ["version_id"])
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("active_index")
    op.drop_table("index_versions")
    op.drop_table("documents")
