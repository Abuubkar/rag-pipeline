"""Chunking with LlamaIndex node parsers.

SentenceSplitter packs whole sentences into chunks of up to `chunk_size` tokens, carrying
`chunk_overlap` tokens into the next chunk so a fact split across a boundary isn't lost.
"""

from importlib.metadata import version

from llama_index.core import Document as LIDocument
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import get_tokenizer

from rag.domain.models import Chunk, Document

LIBRARY = "llamaindex"
LIBRARY_VERSION = version("llama-index-core")


def split_sentence(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    tokenizer = get_tokenizer()  # the same tokenizer the splitter counts with
    chunks: list[Chunk] = []
    for doc in docs:
        # Metadata is attached to chunks by us later, so don't let LlamaIndex mix it into the text.
        nodes = splitter.get_nodes_from_documents([LIDocument(text=doc.body, id_=doc.id)])
        for ordinal, node in enumerate(nodes):
            text = node.get_content()
            chunks.append(
                Chunk(
                    document_id=doc.id,
                    ordinal=ordinal,
                    text=text,
                    token_count=len(tokenizer(text)),
                )
            )
    return chunks
