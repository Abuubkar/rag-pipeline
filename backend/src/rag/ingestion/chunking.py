"""Step 3 of ingestion: split documents into chunks.

The strategy comes from config (`chunking.library` + `chunking.splitter`). Milestone 3 adds
LlamaIndex semantic and LangChain recursive splitters for comparison.
"""

from collections.abc import Callable

from rag.core.config import ChunkingConfig
from rag.domain.models import Chunk, Document
from rag.observability.tracing import annotate, traced
from rag.packages.chunking import langchain, llamaindex

EmbedFn = Callable[[list[str]], list[list[float]]]


def chunker_manifest(cfg: ChunkingConfig) -> dict:
    """What gets recorded in the index manifest for this chunker (only the knobs it uses)."""
    library_version = {"llamaindex": llamaindex.LIBRARY_VERSION, "langchain": langchain.LIBRARY_VERSION}[cfg.library]
    unused = set() if cfg.splitter == "semantic" else {"breakpoint_percentile"}
    return {**cfg.model_dump(exclude=unused), "library_version": library_version}


@traced("ingestion.chunk")
def chunk_documents(docs: list[Document], cfg: ChunkingConfig, embed: EmbedFn) -> list[Chunk]:
    """`embed` is only used by the semantic splitter, which embeds sentences to find topic shifts."""
    match (cfg.library, cfg.splitter):
        case ("llamaindex", "sentence"):
            chunks = llamaindex.split_sentence(docs, cfg.chunk_size, cfg.chunk_overlap)
        case ("llamaindex", "semantic"):
            chunks = llamaindex.split_semantic(
                docs, embed, cfg.breakpoint_percentile, cfg.chunk_size, cfg.chunk_overlap
            )
        case ("langchain", "recursive"):
            chunks = langchain.split_recursive(docs, cfg.chunk_size, cfg.chunk_overlap)
        case _:
            raise ValueError(f"Unknown chunker: {cfg.library}/{cfg.splitter}")

    sizes = [c.token_count for c in chunks]
    annotate(output={
        "chunks": len(chunks),
        "avg_tokens": round(sum(sizes) / max(len(sizes), 1)),
        "max_tokens": max(sizes, default=0),
    })
    return chunks
