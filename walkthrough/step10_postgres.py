"""Step 10: persistent vector store (Postgres + pgvector).

Same idea as step 9, but the vectors live in a Postgres table instead of Python memory, so they
survive restarts: embed once, search many times.

- PGEngine:       the connection pool to Postgres
- PGVectorStore:  LangChain's vector store on top of a Postgres table (replaces the deprecated PGVector)

Each chunk gets a *stable* id derived from its article URL and position, so re-running this
script only embeds chunks that aren't stored yet.

Run:  docker compose up -d   then   uv run walkthrough/step10_postgres.py
"""

import uuid

from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy.exc import ProgrammingError

from step06_load import REVISION, MultiHopRAGLoader
from step07_split import splitter
from step09_vector_store import NomicEmbeddings

DATABASE_URL = "postgresql+psycopg://rag:rag@localhost:5434/rag"
TABLE = "chunks"
VECTOR_SIZE = 768  # nomic-embed-text's output size; the column type must match the model
BATCH_SIZE = 256
# Column names (the library's defaults). The table and the vector store must agree on them,
# so both calls below get them from this one place.
COLUMNS = {
    "id_column": "langchain_id",
    "content_column": "content",
    "embedding_column": "embedding",
    "metadata_json_column": "langchain_metadata",
}

engine = PGEngine.from_connection_string(url=DATABASE_URL)
# init_vectorstore_table has no "create if missing" option: it fails when the table exists,
# and overwrite_existing=True would drop it (and every stored embedding). So: create it once.
try:
    engine.init_vectorstore_table(
        table_name=TABLE,
        vector_size=VECTOR_SIZE,
        **COLUMNS,
    )
except ProgrammingError:  # table already exists → keep it
    pass
vector_store = PGVectorStore.create_sync(
    engine=engine,
    table_name=TABLE,
    embedding_service=NomicEmbeddings(model="nomic-embed-text"),
    **COLUMNS,
)


def chunk_id(chunk) -> str:
    # Same article + same position → same id on every run.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.metadata['url']}#{chunk.metadata['start_index']}"))


if __name__ == "__main__":
    chunks = splitter.split_documents(MultiHopRAGLoader(revision=REVISION).load())
    ids = [chunk_id(chunk) for chunk in chunks]

    stored = {doc.id for doc in vector_store.get_by_ids(ids)}
    missing = [(chunk, id_) for chunk, id_ in zip(chunks, ids) if id_ not in stored]
    print(f"{len(chunks)} chunks · {len(stored)} already in Postgres · {len(missing)} to embed")

    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start : start + BATCH_SIZE]
        vector_store.add_documents([chunk for chunk, _ in batch], ids=[id_ for _, id_ in batch])
        print(f"  stored {start + len(batch)}/{len(missing)}")

    for question in ["Where can I get discounts on Kindle and Echo devices?", "Why is the FTC suing Amazon?"]:
        print(f"\nQ: {question}")
        for chunk, score in vector_store.similarity_search_with_score(question, k=5):
            print(f"  {score:.3f}  {chunk.metadata['title'][:60]}")
