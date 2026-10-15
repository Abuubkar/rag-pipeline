# Milestone 1: what to learn alongside the code

Milestone 1 builds the **naive RAG baseline**: load → clean → chunk → embed → store → search → generate with citations.
Each topic below points to the file where it happens. Read the concept, read the file, then do the exercise.
The order follows the data, so read from top to bottom.

---

## 1. Embeddings and vector similarity (≈ 2 h) — the core idea of RAG
**Learn:** what an embedding is; cosine similarity vs dot product vs L2 distance; why normalising a
vector makes cosine == dot product; dimensions (768 here).
**Code:** `src/rag/packages/embeddings/ollama.py` (`_normalize`, `embed_query`, `embed_documents`)
**Exercise:** in a Python shell, embed "FTX trial" and "Sam Bankman-Fried fraud case" and "NFL Monday Night Football",
then compute cosine similarity between each pair by hand.

## 2. Asymmetric retrieval: query vs document prefixes (≈ 30 min)
**Learn:** why some models (nomic, e5, bge) were trained with task prefixes, and what breaks without them.
**Code:** `_PREFIXES` in `packages/embeddings/ollama.py`. The prefixes are also recorded in the index manifest.
**Exercise (after M2 exists, or by eye now):** remove the prefixes, re-ingest with `--limit 50`, and compare retrieved titles.

## 3. Chunking (≈ 1.5 h)
**Learn:** tokens vs words; chunk size and overlap trade-offs (precision vs context); sentence-aware splitting;
why chunk boundaries can cut a fact in half.
**Code:** `packages/chunking/llamaindex.py` (LlamaIndex `SentenceSplitter`), `ingestion/chunking.py`
**Docs:** LlamaIndex → "Node Parser Modules"
**Exercise:** set `chunk_size: 256` in `config/settings.yaml`, run `uv run main.py ingest --limit 50`, then
`uv run main.py index list`. Notice that you get a **new version id**, and think about why.

## 4. Contextual chunk headers (≈ 30 min)
**Learn:** a chunk that says "He denied the charges" is useless without its headline. Prepending the title/source/date
before embedding (but not storing it) is a cheap form of what Anthropic calls *contextual retrieval*.
**Code:** `ingestion/embedding.py::embedding_text`
**Exercise:** flip `contextual_header: false` and compare which articles `ask` retrieves for the same question.

## 5. pgvector and approximate nearest neighbour search (≈ 2 h)
**Learn:** the `vector` type; operators `<=>` (cosine distance), `<->` (L2), `<#>` (negative inner product);
**HNSW** (graph index; `m`, `ef_construction`, `ef_search` trade recall for speed); exact vs approximate search;
**filtered vector search** and why `hnsw.iterative_scan` exists.
**Code:** `packages/vectorstore/pgvector.py`, `db/models.py` (HNSW + GIN indexes), `db/migrations/versions/0001_initial.py`
**Docs:** github.com/pgvector/pgvector README (Querying, HNSW, Filtering sections)
**Exercise:** `docker exec -it rag-postgres psql -U rag -d rag`, then run
`EXPLAIN ANALYZE` on a vector query. Try it once with `SET enable_indexscan = off;` and compare exact vs HNSW plans.

## 6. Postgres basics you'll use all project (≈ 1 h)
**Learn:** SQLAlchemy 2.0 `select()` style, sessions/transactions, `INSERT … ON CONFLICT` (idempotent writes), JSONB,
generated columns, Alembic migrations.
**Code:** `db/session.py`, `db/index_registry.py`, `alembic.ini`
**Preview of M3:** the `tsv` column is already there. Try
`SELECT title FROM chunks JOIN documents d ON d.id=document_id WHERE tsv @@ websearch_to_tsquery('english','FTX trial') LIMIT 5;`

## 7. Data versioning and idempotency (≈ 45 min) — what makes it "production"
**Learn:** content-addressed ids (hash of the inputs); idempotent pipelines; resumable batch jobs; why
"which model built this index?" must be answerable.
**Code:** `ingestion/indexing.py` (`build_manifest`, `version_id_for`, `chunk_id_for`), `container.py::query_embedder`
**ADR:** `docs/adr/0004-index-versioning.md`
**Exercise:** start `ingest`, press Ctrl-C halfway, run it again, and watch it resume. Then set
`embeddings.model: mxbai-embed-large` and run `ask` to see the guard refuse.

## 8. LangChain core: chat models, messages, callbacks (≈ 1.5 h)
**Learn:** `ChatOllama`, `SystemMessage`/`HumanMessage`, `.invoke()`, `usage_metadata`, **callbacks**
(how Langfuse/LangSmith see LLM calls), runnable `config` (`run_name`, `callbacks`).
**Code:** `packages/llm/ollama.py`, `packages/embeddings/ollama.py`
**Docs:** LangChain docs → Models, Messages, Callbacks

## 9. LangGraph fundamentals (≈ 2 h) — the most important framework here
**Learn:** `StateGraph`, a **state** TypedDict, nodes as functions that return *partial updates*, edges, `START`/`END`,
`compile()`, `invoke()`. M3 adds conditional edges and M5 adds streaming, so learn those concepts now too.
**Code:** `packages/orchestration/graph.py`, `pipeline.py::build_answer_graph`, `domain/state.py`
**Docs:** LangGraph docs → "Graph API overview" (State, Nodes, Edges)
**Exercise:** add a dummy node `debug.print_chunks` between search and context in `pipeline.py` that prints titles and
returns `{}`, then remove it.

## 10. Grounded prompting and citations (≈ 1 h)
**Learn:** instructing the model to use only the sources; numbered citations; an explicit refusal string
("Insufficient information." matches MultiHop-RAG's null answers); why we version prompts.
**Code:** `config/prompts/answer.yaml`, `generation/context.py`, `generation/citations.py`
**Exercise:** ask a null-style question, e.g. *"What did the Mars rover discover in November 2023?"*, and check
whether it refuses. Then try weakening the prompt to see how hallucination creeps in.

## 11. Observability with Langfuse (≈ 1 h)
**Learn:** trace → span → generation; how OpenTelemetry context nests spans; why the LangChain callback handler
links LLM calls (with token counts) to the surrounding span.
**Code:** `observability/tracing.py`, `packages/tracing/langfuse.py`
**Setup:** create a free project at cloud.langfuse.com and put `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` in `backend/.env`.
`ask` then prints a trace link.

## 12. Architecture habits (≈ 30 min)
**Learn:** dependency direction, facades, composition root, ADRs.
**Code:** `[tool.importlinter]` in `backend/pyproject.toml` (run `uv run lint-imports`), `container.py`, `docs/adr/`

---

### Suggested order across a week
| Day | Topics |
|---|---|
| 1 | 1, 2, 3 (embeddings + chunking) |
| 2 | 4, 5 (contextual headers + pgvector/HNSW) |
| 3 | 6, 7 (Postgres + versioning) |
| 4 | 8, 9 (LangChain + LangGraph) |
| 5 | 10, 11, 12 (prompting, Langfuse, architecture) |

### Questions you should be able to answer after M1
1. Why does changing `chunk_size` create a new index version instead of overwriting the old one?
2. What goes wrong if documents are embedded with `search_document:` but queries without `search_query:`?
3. Why can a filtered HNSW query return fewer than k rows, and how does `iterative_scan` fix it?
4. In the answer graph, which node would you change to add re-ranking, and what state key would it read and write?
5. Which file would you touch to swap Ollama for Groq, and which files must *not* change?
