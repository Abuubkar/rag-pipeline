# RAG Pipeline: Project Plan

> This is the living plan. To add something, add it to the **Backlog** (section 13) and, once
> it's agreed, move it into a milestone and write an ADR in `docs/adr/`.

## 1. Goal

A production-style RAG system that is **built for learning**. Each part shows how a real
library is used (LangChain, LangGraph, LlamaIndex, Langfuse, LangSmith, pgvector, RAGAS,
Presidio) and how a production RAG pipeline is put together: ingestion, retrieval,
generation, guardrails, evaluation, observability, streaming UI and agents.

### Principles

1. **One entry point, one story.** `backend/main.py` is the only entry point. `src/rag/pipeline.py` reads
   top to bottom like the architecture diagram.
2. **Folders are capabilities, and `pipeline.py` sets the order.** For example, `retrieval/` says
   *what* a module does, and `pipeline.py` says *when* it runs.
3. **Vendor libraries are only imported in `packages/`**, which is grouped by capability
   (`packages/llm/ollama.py`, `packages/chunking/llamaindex.py`, …).
4. **Free and local-first.** Docker runs Postgres; Ollama runs the models. The only cloud services are free tiers.
5. **Every switch can be measured.** Config choices are recorded on traces and eval runs.
6. **Tests cover AI behaviour only** (graphs, guardrails, retrieval, tracing, eval thresholds).
7. **No empty abstractions.** Code is only written in the milestone that needs it.

## 2. Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 (uv) · TypeScript |
| Generation LLM | Ollama (local), default `gemma4:12b` |
| Judge / stronger tasks | Groq free tier (from Milestone 4) |
| Embeddings | Ollama `nomic-embed-text` (768-dim, needs `search_document:` / `search_query:` prefixes) |
| Database | **Postgres 17 + pgvector** (Docker): vectors, full-text search, app data, LangGraph checkpoints |
| Keyword search | Postgres full-text search (`tsvector`); `pg_textsearch` BM25 is a later option |
| Dataset | **MultiHop-RAG** (`yixuantt/MultiHopRAG`, ODC-BY): 609 news articles, 2,556 questions with gold evidence |
| Frameworks | LlamaIndex (chunking), LangChain (models, prompts, structured output, chunking comparison), LangGraph (orchestration, agent, memory, streaming) |
| PII | Presidio (from Milestone 4) |
| Evals | Custom retrieval/answer metrics + RAGAS (from Milestone 4) |
| Observability | Langfuse Cloud (main) · LangSmith (env toggle, auto-traces LangChain/LangGraph) |
| API / UI | FastAPI + SSE · React + Vite+ + TypeScript + StyleX |
| Out of scope for now | Fine-tuning, PyTorch, Laya/Jev decision models, web search tool |

## 3. Architecture layers

```
ENTRY         main.py (Typer CLI)                    api/ (FastAPI + SSE)
STORY         pipeline.py   ingest · answer graph · agent graph · evaluate
COMPOSITION   container.py  (config → concrete objects; the only "switchboard")
CAPABILITIES  ingestion  query  retrieval  generation  guardrails  evaluation  agents
SUPPORT       observability/ (@traced, annotate, trace)
PACKAGES      vendor facades grouped by capability (the only vendor imports)
DB            db/ (SQLAlchemy models, migrations, index registry)
FOUNDATION    domain/ (pydantic types, graph state, errors)    core/ (config, logging)
```

**Import rules** (enforced by import-linter in `backend/pyproject.toml`, run `uv run lint-imports`):

- Imports only point downward.
- Only `packages/**` may import langchain*, langgraph, llama_index, langfuse, langsmith, ragas
  or presidio.
- Package folders don't import each other; `rag/container.py` wires them together.
- `packages/` returns `domain/` types.
- Capabilities never import `api/` or `pipeline.py`. `pipeline.py` passes runners and functions into
  them instead (e.g. evaluation receives an `answer` callable).
- LangChain message types stay inside `packages/orchestration/` (agent state).

## 4. Repository layout

```
rag-pipeline/
├── docker-compose.yml              # postgres + pgvector (only container)
├── .env.example
├── docs/
│   ├── PLAN.md                     # this file
│   ├── adr/                        # architecture decision records
│   └── concepts/                   # learning notes per milestone
├── backend/
│   ├── pyproject.toml
│   ├── main.py                     # ingest | ask | chat | eval | serve
│   ├── alembic.ini
│   ├── config/
│   │   ├── settings.yaml
│   │   └── prompts/*.yaml          # git = source of truth for prompts
│   ├── src/rag/
│   │   ├── pipeline.py             # THE STORY
│   │   ├── ingestion/              loader · cleaning · chunking · embedding · indexing (· pii in M4)
│   │   ├── query/                  routing · rewriting                         (M3)
│   │   ├── retrieval/              search · reranking · grading · compression  (search in M1)
│   │   ├── generation/             context · generate · citations · delivery
│   │   ├── guardrails/             input · output · tools                      (M4)
│   │   ├── evaluation/             dataset · metrics · experiments · reports   (M2)
│   │   ├── agents/                 tool_loop · agent · tools/                  (M6)
│   │   ├── observability/          tracing.py
│   │   ├── packages/
│   │   │   ├── llm/                ollama.py · groq.py          (LangChain)
│   │   │   ├── embeddings/         ollama.py                    (LangChain)
│   │   │   ├── documents/          multihop_rag.py              (HF datasets)
│   │   │   ├── chunking/           llamaindex.py · langchain.py
│   │   │   ├── vectorstore/        pgvector.py
│   │   │   ├── orchestration/      graph.py · checkpointer.py · streaming.py (LangGraph)
│   │   │   ├── tracing/            langfuse.py · langsmith.py
│   │   │   ├── judging/            ragas.py                     (M4)
│   │   │   └── pii/                presidio.py                  (M4)
│   │   ├── db/                     models.py · session.py · migrations/
│   │   ├── api/                    app.py · sse.py · routes/    (M5)
│   │   ├── container.py            # composition root
│   │   ├── domain/                 models.py · state.py · errors.py (· events.py in M5)
│   │   └── core/                   config.py · logging.py
│   └── tests/                      # AI behaviour only
└── web/                            # React + Vite+ + TS + StyleX (M5)
```

## 5. Workflows (`pipeline.py`)

1. **Ingest:** load → clean → (PII) → chunk → embed → index (versioned)
2. **Answer (LangGraph):**
   `guardrails.input → query.route → query.rewrite → retrieval.search → retrieval.rerank →
   retrieval.grade (retry ≤2) → retrieval.compress → generation.context → generation.generate →
   generation.citations → guardrails.output`
   Milestone 1 builds the minimal version: `retrieval.search → generation.context → generation.generate → generation.citations`.
3. **Agent (LangGraph + a hand-written loop):** the same tools and limits for both.
4. **Evaluate:** dataset → run the answer workflow → metrics → report (stored in Postgres, mirrored to Langfuse).

## 6. Data model (Postgres)

| Table | Purpose |
|---|---|
| `documents` | Source articles (id, title, source, category, published_at, url, body, content_hash) |
| `index_versions` | id = `{splitter}-{sha256(manifest)[:10]}`, manifest jsonb, status (`building`/`ready`/`failed`/`retired`), counts, timestamps, error |
| `active_index` | alias → index_version_id |
| `chunks` | id (deterministic hash), version_id, document_id, ordinal, text, generated `tsv`, `embedding vector(768)`, metadata jsonb |
| `conversations`, `messages`, `feedback` | App state (M5) |
| `eval_runs`, `eval_results`, `judge_cache` | Evaluation (M2/M4) |
| LangGraph checkpoint tables | Conversation memory (`PostgresSaver`) |

**Index versioning.** The manifest records the dataset revision and content hash, the
preprocessing version, the chunker (library, splitter, size, overlap, library version) and the
embedder (provider, model, digest, dims, prefix scheme, normalisation). The same manifest always
produces the same version id, so re-ingesting is a no-op. Chunk ids are deterministic, so retries
are idempotent. **The query-time embedder is built from the active version's manifest**; if config
disagrees with it, the query is refused.

## 7. Retrieval

- **Dense:** pgvector cosine distance (`<=>`) with an HNSW index.
- **Keyword:** Postgres full-text search (`websearch_to_tsquery` + `ts_rank_cd`) (M3).
- **Fusion:** RRF, `score = Σ 1/(60 + rank_i)`. Convex α-fusion over min-max-normalised scores is
  a later option. α = weight of the dense side; α=1 means dense-only.
- **Fairness:** chunkers are compared at an **equal context token budget**.

## 8. Generation, guardrails, delivery

- Citations: numbered sources `[n]` are parsed and checked against the context.
- Delivery modes (M4): `buffered` (validate, then send) vs `stream` (send, then show a verdict).
  Streamed text is never "retracted". A failed check shows a banner, the message is marked
  `flagged`, and it's excluded from memory.
- PII policy: names, places, organisations and dates are **not** PII in a public news corpus. Only
  contact, financial and government identifiers are scrubbed.

## 9. Evaluation

- Golden items are derived from MultiHop-RAG: question, reference answer, question_type,
  evidence doc ids (articles matched by title/source), evidence facts, answerable flag (null queries).
- Stratified **dev/test** split by question_type, pinned dataset revision.
- Metrics:
  - Retrieval: hit@k, recall@k, MRR (article level), evidence-fact recall (chunk text)
  - Answers: normalised exact match / contains-match, null-query refusal accuracy
  - Judge (M4): RAGAS faithfulness, answer correctness
- Baselines: **no retrieval** (does the model already know the answer?), dense-only.
- Reproducibility: config hash, index version, git commit, prompt versions and dataset version are
  stored per run. Judge results are cached. The judge is a different model family from the generator.

## 10. Observability

- `@traced("capability.step")` on every step goes to Langfuse. LangSmith traces
  LangChain/LangGraph automatically when `LANGSMITH_TRACING=true`, in native (non-OTel) mode.
- Every trace records index version, config hash and prompt name@version.

## 11. Milestones

| # | Scope | Done when |
|---|---|---|
| **1** | Postgres + pgvector + Alembic; MultiHop-RAG loader; LlamaIndex sentence chunking; Ollama embeddings with prefixes; index versioning; dense search; LangGraph answer graph (search → context → generate → citations); CLI `ingest` / `ask`; Langfuse tracing (optional if keys are missing) | `ingest` reaches `ready`; re-running is a no-op; `ask` returns a cited answer; a mismatched embedder is refused; the trace shows each node |
| **2** | Golden set + dev/test split; retrieval + answer metrics; no-retrieval baseline; `eval_runs` | Reproducible `eval` with per-question-type breakdown |
| **3** | Full-text search + RRF hybrid; 3 chunkers (LlamaIndex sentence, LlamaIndex semantic, LangChain recursive); query rewrite / multi-query / HyDE; listwise rerank; metadata filters; compression | Comparison report with paired confidence intervals |
| **4** | Guardrails (input/output/PII); delivery modes; red-team set; RAGAS with Groq judge; prompt sync to Langfuse; LangSmith check | Red-team scores; buffered mode never sends unvalidated text |
| **5** | FastAPI SSE + React (Vite+, StyleX): chat, citations, pipeline inspector, feedback | Live node timeline; feedback lands on the right trace |
| **6** | Hand-written tool loop vs LangGraph agent on the same tasks | Comparison report |

## 12. Risks to keep in mind

- **16 GB RAM:** the generator and embedder run side by side in Ollama; keep the generator ≤12B.
- **Contamination:** MultiHop-RAG is late-2023 news; the no-retrieval baseline measures it.
- **Groq free-tier limits** (~30 RPM, ~1k RPD for 70B models): judge only a stratified subset; cache results.
- **Fast-moving libraries** (LangChain/LangGraph 1.x, Langfuse SDK, RAGAS 0.4 `llm_factory`): pin everything in `uv.lock`.
- **Filtered HNSW** in pgvector (filtering on version_id): use iterative index scans (pgvector ≥0.8) or partial indexes.
- **Vite+ / StyleX:** check the unplugin works with Vite+ in M5; plain Vite is the fallback.

## 13. Backlog (future additions)

Add ideas here first, then promote them to a milestone.

- [ ] `pg_textsearch` real BM25 vs Postgres full-text search
- [ ] Qdrant as a second vector store (dedicated vector DB comparison)
- [ ] Convex α-fusion vs RRF
- [ ] Decision models (Laya) for routing / guardrails
- [ ] Sentence-gated streaming delivery mode
- [ ] LlamaIndex query-engine end-to-end baseline
- [ ] Web search tool for the agent
- [ ] Fine-tuning embeddings (PyTorch)
- [ ] Second dataset (HotpotQA) for Wikipedia-style comparison
- [ ] Eval and ingest pages in the UI
- [ ] `main.py index retire/gc` to clean up old versions
- [ ] Physically separate `packages/` into its own uv workspace member (only if facades get reused elsewhere)

## 14. Decision history

- Pinecone Local → dropped (end of life 2026-10-07) → Qdrant considered → **Postgres + pgvector**
  (one service, transactional versioning, replaces SQLite).
- rag-mini-wikipedia → dropped (no gold passage labels) → **MultiHop-RAG**.
- Jev → dropped (not free). Laya → backlog.
- Scope reduced for learning focus: fewer chunkers, fewer delivery modes, no ports or abstract classes.
