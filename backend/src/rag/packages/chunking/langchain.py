"""Chunking with LangChain's RecursiveCharacterTextSplitter.

It tries separators in order (paragraph "\n\n", line "\n", word " ", character "") and only falls
back to a finer one when a piece is still longer than `chunk_size`. So chunks follow paragraphs
where they can, unlike SentenceSplitter, which packs sentences regardless of paragraphs.
Length is counted in cl100k_base tokens, the same tokenizer LlamaIndex counts with.
"""

from importlib.metadata import version

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.domain.models import Chunk, Document

LIBRARY = "langchain"
LIBRARY_VERSION = version("langchain-text-splitters")
_ENCODING = "cl100k_base"


def split_recursive(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=_ENCODING, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    encoding = tiktoken.get_encoding(_ENCODING)
    return [
        Chunk(document_id=doc.id, ordinal=i, text=text, token_count=len(encoding.encode(text)))
        for doc in docs
        for i, text in enumerate(splitter.split_text(doc.body))
    ]


def count_tokens(text: str) -> int:
    """cl100k_base tokens, the unit every chunk's token_count is in."""
    return len(tiktoken.get_encoding(_ENCODING).encode(text))
