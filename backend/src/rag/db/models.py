"""Postgres schema (SQLAlchemy). One database holds documents, chunks + vectors, index versions
and evaluation runs.

The embedding column is fixed at 768 dims (nomic-embed-text). A model with other dims needs
a migration — see docs/adr/0004-index-versioning.md.
"""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIMS = 768


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String)
    category: Mapped[str | None] = mapped_column(String)
    author: Mapped[str | None] = mapped_column(String)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    url: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String)


class IndexVersionRow(Base):
    __tablename__ = "index_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String)  # building | ready | failed | retired
    expected_chunks: Mapped[int | None] = mapped_column(Integer)
    indexed_chunks: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ActiveIndexRow(Base):
    """Points a stable alias ("default") at one ready index version."""

    __tablename__ = "active_index"

    alias: Mapped[str] = mapped_column(String, primary_key=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("index_versions.id"))


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # sha256(version|doc|ordinal)
    version_id: Mapped[str] = mapped_column(ForeignKey("index_versions.id", ondelete="CASCADE"))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMS))
    # Keyword search column, maintained by Postgres itself (used from Milestone 3).
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index("ix_chunks_version", "version_id"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )


class EvalRunRow(Base):
    """One `main.py eval …` run, with everything needed to reproduce or compare it."""

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String)  # retrieval | answers
    mode: Mapped[str] = mapped_column(String)  # dense | rag | closed_book
    split: Mapped[str] = mapped_column(String)  # dev | test
    status: Mapped[str] = mapped_column(String)  # running | done | failed
    questions: Mapped[int] = mapped_column(Integer, default=0)
    index_version_id: Mapped[str] = mapped_column(ForeignKey("index_versions.id"))
    config_hash: Mapped[str] = mapped_column(String)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB)  # settings used (no secrets)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB)  # git commit, prompt, model, dataset…
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # report per question type (JSON keeps key order)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvalResultRow(Base):
    """The score of one golden question in one run (lets you diff two runs question by question)."""

    __tablename__ = "eval_results"

    run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"), primary_key=True)
    item_id: Mapped[str] = mapped_column(String, primary_key=True)
    question_type: Mapped[str] = mapped_column(String)
    scores: Mapped[dict[str, Any]] = mapped_column(JSONB)
