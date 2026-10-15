# 0001. Capability folders, with `pipeline.py` as the story

- Status: accepted
- Date: 2026-10-16

## Context
The project is for learning, so a reader must be able to see the whole RAG flow in one place
and still find each concern in its own module.

## Decision
- Folders are named after **capabilities** (`ingestion/`, `retrieval/`, `generation/`, …).
- `src/rag/pipeline.py` is the only place that defines **order**: the ingest workflow and the
  LangGraph answer graph, with nodes named `capability.step` (e.g. `retrieval.search`).
- `backend/main.py` is the single entry point (CLI). The API (Milestone 5) calls the same functions.

## Alternatives considered
- `phases/p1_…/p6_…` folders: they encode order in folder names, which stops making sense as soon as
  the agent reuses retrieval, and ingestion/evaluation are not steps of answering a question.
- `workflows/` + `capabilities/`: creates two places that tell the story.

## Consequences
Node names line up across the code, the traces and the UI timeline. If `pipeline.py` grows too long, it
can become a `pipeline/` package with one file per workflow.
