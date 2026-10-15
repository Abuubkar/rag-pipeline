"""Plain domain types. Every layer speaks these; no vendor objects cross a package boundary."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Ingestion ────────────────────────────────────────────────────────
class Document(BaseModel):
    """One source article from the corpus."""

    id: str
    title: str
    body: str
    source: str | None = None
    category: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    url: str | None = None
    content_hash: str


class Chunk(BaseModel):
    """A piece of a document, produced by a chunker."""

    document_id: str
    ordinal: int  # position within the document
    text: str
    token_count: int


class EmbeddedChunk(BaseModel):
    """A chunk ready to be stored: deterministic id + dense vector."""

    id: str
    version_id: str
    chunk: Chunk
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Index versioning ─────────────────────────────────────────────────
class EmbedderInfo(BaseModel):
    """Everything that changes the meaning of a vector. Part of the index manifest."""

    provider: str
    model: str
    model_digest: str
    dims: int
    document_prefix: str
    query_prefix: str
    normalize: bool


class IndexManifest(BaseModel):
    """Every input that affects what ends up in the index. Its hash is the version id."""

    schema_version: int = 1
    dataset: dict[str, Any]
    preprocessing: dict[str, Any]
    chunking: dict[str, Any]
    dense: EmbedderInfo


class IndexVersion(BaseModel):
    id: str
    manifest: IndexManifest
    status: str  # building | ready | failed | retired
    expected_chunks: int | None = None
    indexed_chunks: int = 0
    error: str | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None


class IngestionReport(BaseModel):
    version_id: str
    status: str
    documents: int
    chunks: int
    newly_embedded: int
    skipped_existing: int
    seconds: float
    activated: bool


# ── Retrieval & generation ───────────────────────────────────────────
class ScoredChunk(BaseModel):
    """A retrieved chunk with the document fields needed to cite it."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    title: str
    source: str | None = None
    published_at: datetime | None = None
    url: str | None = None


class Citation(BaseModel):
    number: int  # the [n] used in the answer
    chunk_id: str
    document_id: str
    title: str
    source: str | None = None
    url: str | None = None


class Completion(BaseModel):
    text: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)


class Answer(BaseModel):
    question: str
    text: str
    citations: list[Citation]
    invalid_citations: list[int]  # [n] markers that point to no source
    sources: list[ScoredChunk]
    version_id: str
    prompt: str  # name@version
    model: str
