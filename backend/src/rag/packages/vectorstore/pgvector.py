"""Vector storage, dense search (pgvector) and keyword search (Postgres full-text search).

Dense search uses the HNSW index with cosine distance (`<=>`). Because every query filters on
`version_id`, pgvector's iterative index scan is switched on: without it, HNSW returns its
top candidates first and the filter can leave fewer than k rows.

Keyword search uses the generated `tsv` column (GIN index) and ranks with `ts_rank_cd`.
"""

from sqlalchemy import Text, func, select, text
from sqlalchemy.dialects.postgresql import TSQUERY, insert
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
    return [_scored(chunk, doc, 1.0 - float(dist)) for chunk, doc, dist in session.execute(stmt)]  # cosine similarity


def keyword_search(session: Session, version_id: str, query: str, k: int) -> list[ScoredChunk]:
    # plainto_tsquery stems the words and drops stop words, but joins them with AND, so a full
    # question matches almost nothing ("'cryptocurr' & 'trial' & 'fraud' & …"). Swapping & for |
    # matches chunks with ANY of the words; ts_rank_cd then ranks chunks that contain more of
    # them, closer together, first.
    tsquery = func.replace(func.plainto_tsquery("english", query).cast(Text), "&", "|").cast(TSQUERY)
    rank = func.ts_rank_cd(ChunkRow.tsv, tsquery)
    stmt = (
        select(ChunkRow, DocumentRow, rank.label("rank"))
        .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
        .where(ChunkRow.version_id == version_id, ChunkRow.tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(k)
    )
    return [_scored(chunk, doc, float(r)) for chunk, doc, r in session.execute(stmt)]


def _scored(chunk: ChunkRow, doc: DocumentRow, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk.id,
        document_id=doc.id,
        text=chunk.text,
        score=score,
        title=doc.title,
        source=doc.source,
        published_at=doc.published_at,
        url=doc.url,
        token_count=chunk.token_count,
    )
