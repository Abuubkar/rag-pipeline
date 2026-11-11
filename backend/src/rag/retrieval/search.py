"""Find the chunks that answer the question.

Three modes (retrieval.mode in settings.yaml):

dense    embed the question, return the nearest chunk vectors. Finds meaning ("crypto
         exchange collapse" ≈ "FTX bankruptcy") but can miss exact names, numbers and rare words.
keyword  Postgres full-text search. Finds exact words ("Bankman-Fried", "Q3") but not synonyms.
hybrid   run both, then merge the two ranked lists with Reciprocal Rank Fusion (RRF).

RRF only looks at *ranks*, not scores, because the two scores aren't comparable (cosine
similarity ≈ 0.3–0.9, ts_rank_cd ≈ 0.0–10). Each chunk gets Σ 1 / (rrf_k + rank) over the lists
it appears in, so a chunk that is high in both lists beats one that tops only one of them.
rrf_k (60, from the original paper) flattens the curve: rank 1 vs rank 2 matters less.
"""

from typing import Callable

from rag.db.session import session_scope
from rag.domain.models import ScoredChunk
from rag.domain.state import AnswerState
from rag.observability.tracing import annotate
from rag.packages.embeddings.ollama import OllamaEmbedder
from rag.packages.vectorstore import pgvector


def dense_search(
    question: str, *, db_url: str, version_id: str, embedder: OllamaEmbedder, top_k: int
) -> list[ScoredChunk]:
    query_vector = embedder.embed_query(question)
    with session_scope(db_url) as s:
        return pgvector.dense_search(s, version_id, query_vector, top_k)


def keyword_search(question: str, *, db_url: str, version_id: str, top_k: int) -> list[ScoredChunk]:
    with session_scope(db_url) as s:
        return pgvector.keyword_search(s, version_id, question, top_k)


def rrf_fuse(ranked_lists: list[list[ScoredChunk]], rrf_k: int, top_k: int) -> list[ScoredChunk]:
    fused: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1 / (rrf_k + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    best = sorted(fused, key=fused.get, reverse=True)[:top_k]  # type: ignore[arg-type]
    return [chunks[cid].model_copy(update={"score": fused[cid]}) for cid in best]


def search(
    question: str, *, mode: str, db_url: str, version_id: str, embedder: OllamaEmbedder,
    top_k: int, candidates: int, rrf_k: int,
) -> list[ScoredChunk]:
    if mode == "dense":
        return dense_search(question, db_url=db_url, version_id=version_id, embedder=embedder, top_k=top_k)
    if mode == "keyword":
        return keyword_search(question, db_url=db_url, version_id=version_id, top_k=top_k)
    if mode == "hybrid":
        # Each side brings more candidates than we keep, so a chunk ranked 20th by one search
        # but 3rd by the other can still make the final top_k.
        dense = dense_search(question, db_url=db_url, version_id=version_id, embedder=embedder,
                             top_k=max(candidates, top_k))
        keyword = keyword_search(question, db_url=db_url, version_id=version_id,
                                 top_k=max(candidates, top_k))
        return rrf_fuse([dense, keyword], rrf_k, top_k)
    raise ValueError(f"Unknown retrieval mode: {mode} (dense | keyword | hybrid)")


def search_node(**search_kwargs) -> Callable[[AnswerState], dict]:
    """search_kwargs: everything `search` takes except the question."""
    def node(state: AnswerState) -> dict:
        chunks = search(state["question"], **search_kwargs)
        annotate(output=[{"title": c.title, "score": round(c.score, 4)} for c in chunks],
                 metadata={"mode": search_kwargs["mode"]})
        return {"chunks": chunks}

    return node
