"""Load → split → embed and store only the chunks that aren't stored yet. (walkthrough steps 6, 7, 10)"""

import uuid
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_langchain.config import settings
from rag_langchain.loaders import MultiHopRAGLoader
from rag_langchain.vectorstore import vector_store

BATCH_SIZE = 256


def chunk_id(chunk: Document) -> str:
    # Same article + same position → same id on every run, so re-running skips stored chunks.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.metadata['url']}#{chunk.metadata['start_index']}"))


def ingest(log: Callable[[str], None] = print) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(MultiHopRAGLoader(revision=settings.dataset_revision).load())
    ids = [chunk_id(chunk) for chunk in chunks]

    store = vector_store()
    stored = {document.id for document in store.get_by_ids(ids)}
    missing = [(chunk, id_) for chunk, id_ in zip(chunks, ids) if id_ not in stored]
    log(f"{len(chunks)} chunks · {len(stored)} already stored · {len(missing)} to embed")

    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start : start + BATCH_SIZE]
        store.add_documents([chunk for chunk, _ in batch], ids=[id_ for _, id_ in batch])
        log(f"  stored {start + len(batch)}/{len(missing)}")
