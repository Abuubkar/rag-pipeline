"""Step 4 of ingestion: turn chunks into vectors.

With `contextual_header` on, each chunk is embedded together with its article's title,
source and date. A chunk like "He denied the charges." means little alone; with the
headline in front, its vector lands near questions about that story. Only the vector sees
the header — the stored chunk text stays unchanged.
"""

from rag.domain.models import Chunk, Document
from rag.packages.embeddings.ollama import OllamaEmbedder


def embedding_text(chunk: Chunk, doc: Document, contextual_header: bool) -> str:
    if not contextual_header:
        return chunk.text
    date = doc.published_at.date().isoformat() if doc.published_at else "unknown date"
    return f"{doc.title}\n{doc.source or 'unknown source'}, {date}\n\n{chunk.text}"


def embed_texts(embedder: OllamaEmbedder, texts: list[str]) -> list[list[float]]:
    return embedder.embed_documents(texts)
