"""Find the chunks most similar to the question (dense vector search).

Milestone 3 adds keyword search (Postgres full-text) and fuses both lists with RRF.
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


def search_node(
    *, db_url: str, version_id: str, embedder: OllamaEmbedder, top_k: int
) -> Callable[[AnswerState], dict]:
    def node(state: AnswerState) -> dict:
        chunks = dense_search(
            state["question"], db_url=db_url, version_id=version_id, embedder=embedder, top_k=top_k
        )
        annotate(output=[{"title": c.title, "score": round(c.score, 3)} for c in chunks])
        return {"chunks": chunks}

    return node
