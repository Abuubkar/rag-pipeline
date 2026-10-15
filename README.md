# RAG Pipeline (learning project)

A production-style Retrieval-Augmented Generation system built to learn LangChain, LangGraph,
LlamaIndex, pgvector, Langfuse/LangSmith and RAG evaluation, one milestone at a time.

- **Plan & backlog:** [docs/PLAN.md](docs/PLAN.md)
- **Decisions:** [docs/adr/](docs/adr/)
- **What to learn per milestone:** [docs/concepts/](docs/concepts/)
- **Start reading the code at:** `backend/src/rag/pipeline.py`

## Quickstart

Prerequisites: Docker, [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/).

```bash
ollama pull nomic-embed-text && ollama pull gemma4:12b
docker compose up -d                      # Postgres 17 + pgvector on :5433
cd backend
cp ../.env.example .env                   # optional: Langfuse / LangSmith keys
uv sync
uv run main.py db-upgrade                 # create the schema
uv run main.py ingest                     # ~6 min: 609 articles → ~3.4k chunks
uv run main.py ask "Who is the individual in the crypto industry facing a fraud trial?"
uv run main.py index list
uv run pytest                             # AI-behaviour tests
uv run lint-imports                       # architecture rules
```

## Dataset attribution

MultiHop-RAG (ODC-BY): Yixuan Tang and Yi Yang, *MultiHop-RAG: Benchmarking Retrieval-Augmented
Generation for Multi-Hop Queries*, 2024, arXiv:2401.15391.
