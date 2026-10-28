"""Settings, read from environment variables, with defaults for local development."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.environ.get("RAG_DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5434/rag")
    chat_model: str = os.environ.get("RAG_CHAT_MODEL", "gemma4:12b")
    embedding_model: str = os.environ.get("RAG_EMBEDDING_MODEL", "nomic-embed-text")
    vector_size: int = 768  # must match the embedding model
    table: str = "chunks"
    dataset_revision: str = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"  # pinned MultiHop-RAG commit
    chunk_size: int = 1000
    chunk_overlap: int = 200
    k_per_search: int = 5  # chunks fetched by each planned search
    top_n: int = 5  # chunks kept after reranking


settings = Settings()
