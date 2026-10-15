# 0006. Local models: gemma4:12b for answers, nomic-embed-text for vectors

- Status: accepted
- Date: 2026-10-16

## Context
Everything should be free, and the dev machine is an M1 Pro with 16 GB RAM.

## Decision
- Generator: `gemma4:12b` via Ollama (7.6 GB). `qwen3.6` (23 GB) doesn't fit in memory next to the
  embedder and Postgres. Temperature 0, fixed seed, `reasoning=False`, `num_ctx` 8192.
- Embeddings: `nomic-embed-text` (768 dims) with its **required** task prefixes
  (`search_document: ` / `search_query: `), L2-normalised.
- Chunks are embedded with a contextual header (title, source, date); stored text stays unchanged.
- Groq (free tier) joins in M4 as a different-family eval judge.

## Consequences
The first call after idle takes ~30–40 s while Ollama loads the model. Swapping models is a config change
plus re-ingest (for embeddings), which the version guard enforces.
