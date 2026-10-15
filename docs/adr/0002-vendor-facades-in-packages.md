# 0002. Vendor libraries only inside `packages/`, grouped by capability

- Status: accepted
- Date: 2026-10-16

## Context
The goal is to learn LangChain, LangGraph, LlamaIndex, Langfuse and friends — and to see clearly
*where* each one is used and what it is responsible for.

## Decision
- Only `rag/packages/**` imports AI vendor libraries. Folders are capabilities
  (`llm/`, `embeddings/`, `chunking/`, `vectorstore/`, `orchestration/`, `tracing/`, …) and
  file names are vendors (`chunking/llamaindex.py`, `llm/ollama.py`).
- Facades return `domain/` types, never library objects.
- Package folders don't import each other; `rag/container.py` wires them (e.g. it hands Langfuse's
  LangChain callback to the chat model).
- Enforced by import-linter contracts in `backend/pyproject.toml` (`uv run lint-imports`).
- No abstract base classes until a second implementation exists.

## Framework responsibilities
| Concern | Owner |
|---|---|
| Chunking | LlamaIndex node parsers (LangChain splitters added for comparison in M3) |
| Chat models, embeddings, messages, callbacks | LangChain (`langchain-ollama`) |
| Control flow, state, (later) memory/streaming/agent | LangGraph |
| Retrieval SQL | our code + pgvector |
| Tracing | Langfuse |

## Consequences
Comparing two vendors for the same job means adding one file next to the other. Infra libraries
(SQLAlchemy, Typer, pydantic, PyYAML) are used directly — wrapping them teaches nothing.
