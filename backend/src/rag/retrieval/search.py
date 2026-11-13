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
from rag.domain.models import ScoredChunk, SearchQueries
from rag.domain.state import AnswerState
from rag.observability.tracing import annotate
from rag.packages.embeddings.ollama import OllamaEmbedder
from rag.packages.vectorstore import pgvector
from rag.retrieval import metadata_filter


def dense_search(
    vector: list[float], *, db_url: str, version_id: str, top_k: int, sources: list[str] | None = None
) -> list[ScoredChunk]:
    with session_scope(db_url) as s:
        return pgvector.dense_search(s, version_id, vector, top_k, sources)


def keyword_search(
    question: str, *, db_url: str, version_id: str, top_k: int, sources: list[str] | None = None
) -> list[ScoredChunk]:
    with session_scope(db_url) as s:
        return pgvector.keyword_search(s, version_id, question, top_k, sources)


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
    searched: SearchQueries | str, *, mode: str, db_url: str, version_id: str, embedder: OllamaEmbedder,
    top_k: int, candidates: int, rrf_k: int, filter_mode: str = "none",
) -> list[ScoredChunk]:
    """`searched` is the question itself, or what a query transform made of it (several queries,
    HyDE passages). Every query and passage gives one ranked list; more than one list → RRF.
With a metadata filter, every list only searches the publications the queries name."""
    if mode not in ("dense", "keyword", "hybrid"):
        raise ValueError(f"Unknown retrieval mode: {mode} (dense | keyword | hybrid)")
    if isinstance(searched, str):
        searched = SearchQueries(queries=[searched])
    # Each list brings more candidates than we keep, so a chunk ranked 20th in one list
    # but 3rd in another can still make the final top_k.
    n = max(candidates, top_k)
    sources = metadata_filter.sources_for(" ".join(searched.queries), mode=filter_mode, db_url=db_url)
    annotate(metadata={"sources": sources})
    where = dict(db_url=db_url, version_id=version_id, top_k=n, sources=sources)
    ranked_lists = []
    for query in searched.queries:
        if mode in ("dense", "hybrid"):
            ranked_lists.append(dense_search(embedder.embed_query(query), **where))
        if mode in ("keyword", "hybrid"):
            ranked_lists.append(keyword_search(query, **where))
    for passage in searched.passages:  # HyDE: a passage is compared with chunks as a document
        ranked_lists.append(dense_search(embedder.embed_documents([passage])[0], **where))
    if len(ranked_lists) == 1:
        return ranked_lists[0][:top_k]  # one list: keep its own scores (cosine or ts_rank_cd)
    return rrf_fuse(ranked_lists, rrf_k, top_k)


def search_node(**search_kwargs) -> Callable[[AnswerState], dict]:
    """search_kwargs: everything `search` takes except what to search for."""
    def node(state: AnswerState) -> dict:
        chunks = search(state.get("searched") or state["question"], **search_kwargs)
        annotate(output=[{"title": c.title, "score": round(c.score, 4)} for c in chunks],
                 metadata={"mode": search_kwargs["mode"]})
        return {"chunks": chunks}

    return node
