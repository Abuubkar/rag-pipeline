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
    token_count: int = 0


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


# ── Evaluation ───────────────────────────────────────────────────────
class GoldenItem(BaseModel):
    """One evaluation question with its reference answer and the articles that hold the evidence."""

    id: str
    question: str
    answer: str
    question_type: str  # inference_query | comparison_query | temporal_query | null_query
    evidence_document_ids: list[str]  # gold articles (empty for null queries)
    evidence_facts: list[str]  # the exact sentences the answer relies on
    answerable: bool  # False for null queries: the right answer is "Insufficient information."
    split: str = ""  # dev | test, set by evaluation.dataset


class RetrievalScore(BaseModel):
    """How well retrieval did on one golden question. Keys of the dicts are k (top-k chunks)."""

    item_id: str
    question_type: str
    hit: dict[int, bool]  # at least one gold article in the top k
    recall: dict[int, float]  # share of gold articles in the top k
    reciprocal_rank: float  # 1 / rank of the first chunk from a gold article (0 if none)
    fact_recall: float  # share of evidence sentences found word-for-word in the top-k chunks
    context_tokens: int = 0  # tokens in those top-k chunks: what the LLM would have to read


class AnswerScore(BaseModel):
    """How the generated answer to one golden question compares with the reference answer."""

    item_id: str
    question_type: str
    answer: str  # what the model said
    reference: str  # the dataset's answer
    exact_match: bool  # first sentence == reference (after normalising)
    correct: bool  # reference found in the first sentence; for null queries: the model refused
    refused: bool  # model said "Insufficient information."
    cited: bool  # at least one valid [n] citation
    invalid_citations: int


class EvalRun(BaseModel):
    """One evaluation run. Same config_hash + index + git commit should give the same numbers."""

    id: str
    kind: str  # retrieval | answers
    mode: str  # dense | keyword | hybrid | rag | closed_book
    split: str
    status: str  # running | done | failed
    label: str | None = None  # set → the run appears in docs/results
    questions: int = 0
    index_version_id: str
    config_hash: str
    config: dict[str, Any]
    provenance: dict[str, Any]  # git_commit, git_dirty, prompt, model, model_digest, dataset_revision, limit
    summary: dict[str, dict[str, int | float]] | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None
