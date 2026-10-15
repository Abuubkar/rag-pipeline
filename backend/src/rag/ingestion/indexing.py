"""Step 5 of ingestion: write chunks + vectors into a *versioned* index.

Why versions? A vector only makes sense next to the exact model, prefixes, chunker and
cleaning rules that produced it. The manifest lists all of them; its hash is the version id.

- Same inputs → same version id → re-running ingest is a no-op.
- Chunk ids are hash(version, document, position) → a crashed run resumes where it stopped.
- Status goes building → ready (all chunks verified) or failed (error recorded).
"""

import hashlib
import json
from typing import Callable

from rag.core.config import ChunkingConfig, DatasetConfig
from rag.db import index_registry
from rag.db.session import session_scope
from rag.domain.models import Chunk, Document, EmbeddedChunk, EmbedderInfo, IndexManifest
from rag.ingestion.cleaning import CLEANING_VERSION
from rag.ingestion.chunking import chunker_manifest
from rag.ingestion.embedding import embed_texts, embedding_text
from rag.observability.tracing import annotate, traced
from rag.packages.embeddings.ollama import OllamaEmbedder
from rag.packages.vectorstore import pgvector

ProgressFn = Callable[[int, int], None]


def build_manifest(
    dataset: DatasetConfig, docs: list[Document], chunking: ChunkingConfig, dense: EmbedderInfo
) -> IndexManifest:
    corpus_hash = hashlib.sha256(
        "".join(f"{d.id}:{d.content_hash}" for d in sorted(docs, key=lambda d: d.id)).encode()
    ).hexdigest()
    return IndexManifest(
        dataset={"name": dataset.name, "revision": dataset.revision,
                 "documents": len(docs), "content_hash": corpus_hash},
        preprocessing={"cleaning_version": CLEANING_VERSION},
        chunking=chunker_manifest(chunking),
        dense=dense,
    )


def version_id_for(manifest: IndexManifest) -> str:
    canonical = json.dumps(manifest.model_dump(mode="json"), sort_keys=True)
    return f"{manifest.chunking['splitter']}-{hashlib.sha256(canonical.encode()).hexdigest()[:10]}"


def chunk_id_for(version_id: str, chunk: Chunk) -> str:
    return hashlib.sha256(f"{version_id}|{chunk.document_id}|{chunk.ordinal}".encode()).hexdigest()[:32]


@traced("ingestion.index")
def index_chunks(
    db_url: str,
    version_id: str,
    manifest: IndexManifest,
    chunks: list[Chunk],
    docs: list[Document],
    embedder: OllamaEmbedder,
    batch_size: int,
    contextual_header: bool,
    on_progress: ProgressFn | None = None,
) -> tuple[int, int]:
    """Embed and store every chunk not already stored. Returns (newly_embedded, skipped)."""
    docs_by_id = {d.id: d for d in docs}

    with session_scope(db_url) as s:
        index_registry.start_building(s, version_id, manifest, expected=len(chunks))
        existing = pgvector.existing_chunk_ids(s, version_id)

    pending = [c for c in chunks if chunk_id_for(version_id, c) not in existing]
    done = len(chunks) - len(pending)
    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            texts = [embedding_text(c, docs_by_id[c.document_id], contextual_header) for c in batch]
            vectors = embed_texts(embedder, texts)
            rows = [
                EmbeddedChunk(
                    id=chunk_id_for(version_id, c),
                    version_id=version_id,
                    chunk=c,
                    embedding=v,
                    metadata=_chunk_metadata(docs_by_id[c.document_id]),
                )
                for c, v in zip(batch, vectors)
            ]
            with session_scope(db_url) as s:  # one transaction per batch = resumable
                pgvector.insert_chunks(s, rows)
                done += len(rows)
                index_registry.record_progress(s, version_id, done)
            if on_progress:
                on_progress(done, len(chunks))

        _verify_and_finish(db_url, version_id, expected=len(chunks))
    except Exception as exc:
        with session_scope(db_url) as s:
            index_registry.finish(s, version_id, "failed", error=repr(exc))
        raise

    annotate(output={"version_id": version_id, "embedded": len(pending), "skipped": len(existing)})
    return len(pending), len(chunks) - len(pending)


def _chunk_metadata(doc: Document) -> dict:
    return {
        "title": doc.title,
        "source": doc.source,
        "category": doc.category,
        "published_at": doc.published_at.isoformat() if doc.published_at else None,
    }


def _verify_and_finish(db_url: str, version_id: str, expected: int) -> None:
    with session_scope(db_url) as s:
        stored = pgvector.count_chunks(s, version_id)
        if stored != expected:
            raise RuntimeError(f"Index {version_id}: expected {expected} chunks, found {stored}")
        index_registry.finish(s, version_id, "ready")
