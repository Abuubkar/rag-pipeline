# 0003. Postgres + pgvector as the single database

- Status: accepted (supersedes the Pinecone Local and SQLite plans)
- Date: 2026-10-16

## Context
Pinecone Local reached end of life on 2026-10-07 and its replacement is a stateless mock, so it can't
be used for retrieval tests. We also needed somewhere to store app state, index versions and (later)
LangGraph checkpoints.

## Decision
One Postgres 17 instance with pgvector (`pgvector/pgvector:pg17`, docker-compose, port 5433) stores:
documents, chunks + `vector(768)` embeddings (HNSW, cosine), a generated `tsvector` column for keyword
search, index versions, and later conversations, eval runs and LangGraph checkpoints.

## Alternatives considered
- **Pinecone Starter (cloud):** real Pinecone, but not local, with egress/namespace limits.
- **Qdrant (local):** excellent built-in hybrid search, but a second store to keep in sync with app data.
- **SQLite + a vector DB:** more files and more moving parts.

## Consequences
- Index versioning is transactional: chunks and their status change together.
- Hybrid search (M3) is hand-written SQL + RRF. That's more to write, but it's visible.
- Keyword ranking is `ts_rank`, not true BM25; `pg_textsearch` is in the backlog.
- Queries filter on `version_id`, so HNSW uses `hnsw.iterative_scan = relaxed_order` to still return k rows.
- The embedding column is fixed at 768 dims; a model with other dims needs a migration.
