# Learning Roadmap: RAG with LangChain only

One step = one small feature = one commit. Each step lists the concept, the LangChain piece
that does it, and the package it needs. Packages are installed **only when that step starts**.

Status: ✅ done · 🔜 next · ⬜ planned

The step scripts and their notes live in [`walkthrough/`](walkthrough/). Run them from the repo root,
e.g. `uv run walkthrough/step12_rag_chain.py`. Which LangChain piece to use when: [`LADDER.md`](LADDER.md).

**Production pipeline:** [`src/rag_langchain/`](src/rag_langchain/) combines the preferred pieces of steps 6–20 into
one package with a single entry point:

```bash
uv run rag ingest              # load, split, embed and store (skips stored chunks)
uv run rag ask "question"      # plan searches → filtered search → rerank → answer with citations
uv run rag chat                # follow-up questions in one conversation
```

---

## Part A: Talking to an LLM

| # | Step | You learn | LangChain piece | Package | Status |
|---|---|---|---|---|---|
| 0 | Project setup | Working rules, empty project | — | — | ✅ |
| 1 | Call a local LLM | What a chat model is, messages (system/human/AI), `invoke` | `ChatOllama` | `langchain-ollama` | ✅ |
| 2 | Prompt templates | Reusable prompts with variables | `ChatPromptTemplate` | (comes with step 1) | ✅ |
| 3 | Chains (LCEL) | Joining steps with `\|`: prompt → model → parser | Runnables, `StrOutputParser` | — | ✅ |
| 4 | Streaming | Showing tokens as they're generated | `.stream()` | — | ✅ |
| 5 | Structured output | Getting typed data (JSON) back from an LLM | `with_structured_output` | `pydantic` (declared; already installed) | ✅ |

## Part B: Indexing (preparing documents so they can be searched)

| # | Step | You learn | LangChain piece | Package | Status |
|---|---|---|---|---|---|
| 6 | Load documents | What a `Document` is (text + metadata); how every loader works (`lazy_load`/`load`) | Our own `BaseLoader` (langchain-core) | `huggingface-hub` | ✅ |
| 7 | Split into chunks | Why we chunk; chunk size and overlap | `RecursiveCharacterTextSplitter` | `langchain-text-splitters` | ✅ |
| 8 | Embeddings | Turning text into vectors; cosine similarity | `OllamaEmbeddings` | (step 1 package) | ✅ |
| 9 | In-memory vector store | Storing vectors and searching by similarity | `InMemoryVectorStore` | `numpy` | ✅ |
| 10 | Persistent vector store | Keeping vectors in a real database | `PGVectorStore` + Postgres in Docker | `langchain-postgres`, `sqlalchemy` | ✅ |

## Part C: Retrieval and answering (the core RAG loop)

| # | Step | You learn | LangChain piece | Package | Status |
|---|---|---|---|---|---|
| 11 | Retrievers | The retriever interface; top-k | `as_retriever()` (similarity, MMR) | — | ✅ |
| 12 | First RAG chain | Retrieve → put chunks in prompt → answer | LCEL chain | — | ✅ |
| 13 | Sources and citations | Returning which chunks were used | `RunnablePassthrough.assign` | — | ✅ |
| 14 | Keyword search | Why exact words still matter | Postgres full-text search via `PGVectorStore` + `HybridSearchConfig` (instead of `BM25Retriever`, which is in sunset `langchain-community`) | — | ✅ |
| 15 | Hybrid search | Combining keyword + vector results; reciprocal rank fusion | `PGVectorStore` hybrid search (instead of legacy `EnsembleRetriever`) | — | ✅ |
| 16 | Multi-query | Asking the question several ways | LCEL chain: LLM writes queries → `retriever.map()` → merge, plus the original question (instead of legacy `MultiQueryRetriever`) | — | ✅ |
| 17 | Reranking | Letting the LLM reorder results | LCEL chain: LLM scores each chunk 0–10 with `.batch` + structured output, keep the top 5 (instead of legacy `ContextualCompressionRetriever`) | — | ✅ |
| 18 | Metadata filtering | "Only TechCrunch articles from October" | LLM plans (query, source) searches with structured output → `PGVectorStore` `filter=` (instead of legacy `SelfQueryRetriever`) | — | ✅ |

## Part D: Conversations

| # | Step | You learn | LangChain piece | Package | Status |
|---|---|---|---|---|---|
| 19 | Chat history | Passing previous turns to the model | `MessagesPlaceholder` + a plain list (`InMemoryChatMessageHistory` is deprecated) | — | ✅ |
| 20 | Follow-up questions | Rewriting "what about him?" into a full question | An LLM rewrite step before retrieval, built with LCEL (the old `create_history_aware_retriever` is in legacy `langchain-classic`) | — | ✅ |
| 21 | Saved history | Keeping chats across restarts | Our own `chat_messages` table: one row per message, `messages_to_dict` / `messages_from_dict` (not the deprecated `PostgresChatMessageHistory`) | — | ✅ |
| 21b | Saved history, the v1 way | The same, with LangGraph | A LangGraph checkpointer (`PostgresSaver`) per `thread_id` | `langgraph`, `langgraph-checkpoint-postgres` | ✅ |

## Part E: Safety and quality

| # | Step | You learn | LangChain piece | Package | Status |
|---|---|---|---|---|---|
| 22 | Routing + input guardrail | Sending news questions to RAG, conversation questions to the history, and off-topic or injection attempts to a refusal | Structured-output router + `RunnableBranch` | — | ✅ |
| 23 | Grounded answers | Checking the answer is supported by sources | LLM-as-judge chain | — | ⬜ |
| 24 | Evaluation | Measuring retrieval and answers on gold questions | Plain script over the chains | — | ⬜ |

## Part F: Agents

| # | Step | You learn | LangChain piece | Package | Status |
|---|---|---|---|---|---|
| 25 | Tools | How an LLM "calls a function" | `@tool`, `bind_tools` | — | ⬜ |
| 26 | Agentic RAG | The LLM decides when and what to search | `create_agent` + retriever tool | `langchain` | ⬜ |

## Part G: Seeing inside

| # | Step | You learn | LangChain piece | Package | Status |
|---|---|---|---|---|---|
| 27 | Callbacks | Watching every step a chain runs | Callback handlers | — | ⬜ |
| 28 | Decision point | Is LangChain's own visibility enough, or do we need LangSmith / Langfuse? | — | — | ⬜ |

---

## LangChain limits found along the way

Filled in as we hit them. These are the reasons to reach for LangGraph, LangSmith or Langfuse later.

| Step | Limit | What would solve it |
|---|---|---|
| 5 | `ChatOllama.with_structured_output` defaults to `method="json_schema"`, which constrains the format but never shows the model the field descriptions. You get valid but wrong values, silently. | Use `method="function_calling"`; always check values, not just types |
| 6 | `langchain-community` (home of most classic loaders and retrievers) was sunset on 2026-05-22 and its repo archived. It still installs, but gets no fixes. | Standalone `langchain-<provider>` packages, or your own `BaseLoader` |
| 6 | The old `HuggingFaceDatasetLoader` (a) can't pin a dataset `revision`, (b) needs `datasets` but doesn't declare it (fails only at `.load()`), and (c) runs every text through `json.dumps`, silently turning articles into escaped JSON strings (quotes, literal `\n`, `\u2019`). | A small custom `BaseLoader` with `hf_hub_download(revision=…)` |
| 7 | `RecursiveCharacterTextSplitter`'s `chunk_overlap` is not guaranteed: it only carries over whole pieces, so when a cut lands after a paragraph longer than the overlap, nothing repeats (only 48% of neighbouring chunks overlapped here). It also leaves tiny fragments (53 chunks under 100 characters). | Know it's best-effort; check overlap and tiny chunks on your own data |
| 8 | `OllamaEmbeddings` doesn't add model-specific task prefixes (`search_document:` / `search_query:` for nomic-embed-text), and has no separate setting for document vs query text. Forgetting them silently lowers retrieval quality. | Add the prefixes yourself, the same way at indexing and query time |
| 9 | `InMemoryVectorStore` (in `langchain-core`) needs `numpy` to search, but `langchain-core` doesn't install it: it fails with an ImportError at the first search. | Add `numpy` yourself |
| 9 | `OllamaEmbeddings.embed_query` calls `self.embed_documents`, so a subclass that overrides `embed_documents` also changes every query: our queries silently got both prefixes (`search_document: search_query: …`). | In the subclass, have query methods call `super().embed_documents` directly |
| 10 | `PGVectorStore`'s sync methods (`add_documents`, `similarity_search`) call the **async** embedding methods (`aembed_documents` / `aembed_query`) internally, so an `Embeddings` subclass that overrides only the sync methods is skipped. No error: all 8,448 chunks were stored without the prefix, and found only in step 15 because hand-computed scores didn't match. | Override both sync and async methods; check a stored vector against a known embedding |
| 10 | `PGVector` is deprecated (since `langchain-postgres` 0.0.14); the replacement is `PGVectorStore`. | Use `PGEngine` + `PGVectorStore` |
| 10 | `init_vectorstore_table` has no "create if missing": it errors if the table exists, and `overwrite_existing=True` drops every stored embedding. | Catch `ProgrammingError` and keep the table |
| 10 | `similarity_search_with_score` means different things per store: `InMemoryVectorStore` returns similarity (higher = better), `PGVectorStore` returns cosine distance (lower = better). | Check each store's score meaning before comparing or thresholding |
| 14 | `BM25Retriever` still lives only in the sunset `langchain-community` (`langchain-classic` just redirects to it). | Use the vector store's own keyword search (Postgres full-text search here) |
| 14 | `PGVectorStore` writes the first query into the `HybridSearchConfig` it's given and reuses it, so with a shared config every later search returns the first query's results. No error, just wrong results. | Pass a new `HybridSearchConfig` on every call |
| 14 | No keyword-only mode: it's half of hybrid search. The vector half is turned off with `primary_top_k=0`, but the query is still embedded. | Accept the wasted embedding call, or write the SQL yourself |
| 14 | Keyword matching uses `plainto_tsquery`, which requires **every** word, and it can't be changed. Long natural questions get 0 results. | Keyword search for short, name-heavy queries; combine with vector search (step 15) |
| 15 | The ready-made retrievers (`EnsembleRetriever`, `MultiQueryRetriever`, `ContextualCompressionRetriever`, `SelfQueryRetriever`) moved to the legacy `langchain-classic` in v1, and the current docs don't mention them. v1 has no replacements: it points to vector-store features, plain chains, agents and LangGraph. | Use the store's own hybrid search; build the rest as small LCEL chains |
| 15 | Hybrid isn't automatically better. RRF trusts both lists equally and only looks at ranks, so neither search judges whether a chunk actually answers the question. | Tune `primary_top_k`/`secondary_top_k`, or rerank afterwards (step 17) |
| 16 | `with_structured_output(method="function_calling")` only accepts the exact class name as the tool name. gemma4 sometimes calls `search_queries` instead of `SearchQueries` (2 of 8 dataset questions), and LangChain raises `OutputParserException` instead of matching it. A retry at temperature 0 repeats the same mistake. | Wrap the step in `.with_fallbacks([...], exceptions_to_handle=(OutputParserException,))` |
| 19 | LangChain's chat-history helpers are deprecated: `RunnableWithMessageHistory` (since 1.3.3) and `InMemoryChatMessageHistory` / `BaseChatMessageHistory` (since 1.6.4), all to be removed in 2.0. The docs point to LangGraph persistence (`create_agent` + a checkpointer). LangChain alone no longer has a supported way to store chat history. | A plain list + `MessagesPlaceholder`; a LangGraph checkpointer to persist |
| 20 | `create_history_aware_retriever` moved to the legacy `langchain-classic` package in v1. It is only a branch (no history → search the question; else prompt → LLM → search), so core Runnables rebuild it in a few lines. | A rewrite chain + a plain function that skips it on the first turn |
| 21 | `PostgresChatMessageHistory` (langchain-postgres) builds on the deprecated `BaseChatMessageHistory`, so LangChain alone has no supported class to save chat history. | Our own table with `messages_to_dict` / `messages_from_dict`; a LangGraph checkpointer (step 21b) |

## Notes

- This order is a plan, not a contract. Steps can be split, merged or dropped as we go.
- Steps 2–5, 9 and 11–22 (except 21b) need no new packages, because they use what earlier steps installed.
