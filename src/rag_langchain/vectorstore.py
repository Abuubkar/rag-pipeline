"""The Postgres vector store (pgvector). (walkthrough step 10)"""

from functools import cache

from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from rag_langchain.config import settings
from rag_langchain.embeddings import embeddings

# Column names (the library's defaults). The table and the store must agree on them.
COLUMNS = {
    "id_column": "langchain_id",
    "content_column": "content",
    "embedding_column": "embedding",
    "metadata_json_column": "langchain_metadata",
}


@cache
def vector_store() -> PGVectorStore:
    engine = PGEngine.from_connection_string(url=settings.database_url)
    # init_vectorstore_table has no "create if missing": it fails when the table exists.
    try:
        engine.init_vectorstore_table(table_name=settings.table, vector_size=settings.vector_size, **COLUMNS)
    except ProgrammingError:  # table already exists → keep it
        pass
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=settings.table,
        embedding_service=embeddings(),
        **COLUMNS,
    )


@cache
def publication_names() -> tuple[str, ...]:
    """Every distinct `source` stored in the table (the values a metadata filter can use)."""
    sql = text(f"SELECT DISTINCT langchain_metadata->>'source' FROM {settings.table} ORDER BY 1")
    with create_engine(settings.database_url).connect() as connection:
        return tuple(row[0] for row in connection.execute(sql))
