"""Chunking with LlamaIndex node parsers.

SentenceSplitter packs whole sentences into chunks of up to `chunk_size` tokens, carrying
`chunk_overlap` tokens into the next chunk so a fact split across a boundary isn't lost.

SemanticSplitterNodeParser embeds every sentence (with its neighbours) and cuts where the
similarity between neighbours drops below the `breakpoint_percentile` of the document's
similarities, i.e. where the topic changes. Chunk size is not controlled, so any chunk longer
than `chunk_size` is re-split with SentenceSplitter (the embedder would truncate it otherwise).
"""

from collections.abc import Callable
from importlib.metadata import version

from llama_index.core import Document as LIDocument
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.utils import get_tokenizer

from rag.domain.models import Chunk, Document

LIBRARY = "llamaindex"
LIBRARY_VERSION = version("llama-index-core")

EmbedFn = Callable[[list[str]], list[list[float]]]


class _Embedding(BaseEmbedding):
    """Lets LlamaIndex call our embedder (the container's), so both index and splitter use one model."""

    embed: EmbedFn

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self.embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


def split_sentence(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [chunk for doc in docs for chunk in _to_chunks(doc, _split(splitter, doc.body))]


def split_semantic(
    docs: list[Document], embed: EmbedFn, breakpoint_percentile: int, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    semantic = SemanticSplitterNodeParser(
        embed_model=_Embedding(embed=embed, embed_batch_size=32),
        buffer_size=1,  # compare each sentence together with its neighbours, not alone
        breakpoint_percentile_threshold=breakpoint_percentile,
    )
    fallback = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    tokenizer = get_tokenizer()
    chunks: list[Chunk] = []
    for doc in docs:
        texts = []
        for text in _split(semantic, doc.body):
            texts += _split(fallback, text) if len(tokenizer(text)) > chunk_size else [text]
        chunks += _to_chunks(doc, texts)
    return chunks


def _split(splitter, text: str) -> list[str]:
    # Metadata is attached to chunks by us later, so don't let LlamaIndex mix it into the text.
    return [n.get_content() for n in splitter.get_nodes_from_documents([LIDocument(text=text)])]


def _to_chunks(doc: Document, texts: list[str]) -> list[Chunk]:
    tokenizer = get_tokenizer()  # the same tokenizer the splitters count with
    return [
        Chunk(document_id=doc.id, ordinal=i, text=t, token_count=len(tokenizer(t)))
        for i, t in enumerate(t for t in texts if t.strip())
    ]
