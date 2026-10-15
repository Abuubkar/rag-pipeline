"""Vector storage and dense search with Postgres + pgvector.

Search uses the HNSW index with cosine distance (`<=>`). Because every query filters on
`version_id`, pgvector's iterative index scan is switched on: without it, HNSW returns its
top candidates first and the filter can leave fewer than k rows.
"""

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from rag.db.models import ChunkRow, DocumentRow
from rag.domain.models import EmbeddedChunk, ScoredChunk


def existing_chunk_ids(session: Session, version_id: str) -> set[str]:
    rows = session.scalars(select(ChunkRow.id).where(ChunkRow.version_id == version_id))
    return set(rows)


def count_chunks(session: Session, version_id: str) -> int:
    return session.scalar(
        select(func.count()).select_from(ChunkRow).where(ChunkRow.version_id == version_id)
    ) or 0


def insert_chunks(session: Session, chunks: list[EmbeddedChunk]) -> None:
    """Idempotent: chunk ids are deterministic, so a retried batch is simply skipped."""
    if not chunks:
        return
    stmt = insert(ChunkRow).values(
        [
            {
                "id": c.id,
                "version_id": c.version_id,
                "document_id": c.chunk.document_id,
                "ordinal": c.chunk.ordinal,
                "text": c.chunk.text,
                "token_count": c.chunk.token_count,
                "embedding": c.embedding,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
    )
    session.execute(stmt.on_conflict_do_nothing(index_elements=["id"]))


def dense_search(
    session: Session, version_id: str, query_vector: list[float], k: int
) -> list[ScoredChunk]:
    session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    session.execute(text("SET LOCAL hnsw.ef_search = 100"))

    distance = ChunkRow.embedding.cosine_distance(query_vector)
    stmt = (
        select(ChunkRow, DocumentRow, distance.label("distance"))
        .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
        .where(ChunkRow.version_id == version_id)
        .order_by(distance)
        .limit(k)
    )
    results = []
    for chunk, doc, dist in session.execute(stmt):
        results.append(
            ScoredChunk(
                chunk_id=chunk.id,
                document_id=doc.id,
                text=chunk.text,
                score=1.0 - float(dist),  # cosine similarity
                title=doc.title,
                source=doc.source,
                published_at=doc.published_at,
                url=doc.url,
            )
        )
    return results
