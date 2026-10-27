# Learning Roadmap: RAG with LangChain only

One step = one small feature = one commit. Each step lists the concept, the LangChain piece
that does it, and the package it needs. Packages are installed **only when that step starts**.

Status: ✅ done · 🔜 next · ⬜ planned

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
| 12 | First RAG chain | Retrieve → put chunks in prompt → answer | LCEL chain | — | 🔜 |
| 13 | Sources and citations | Returning which chunks were used | `RunnablePassthrough.assign` | — | ⬜ |
| 14 | Keyword search | Why exact words still matter (BM25) | `BM25Retriever` (⚠️ lives in sunset `langchain-community`; recheck docs first) | `rank-bm25` | ⬜ |
| 15 | Hybrid search | Combining keyword + vector results | `EnsembleRetriever` | `langchain-classic` | ⬜ |
| 16 | Multi-query | Asking the question several ways | `MultiQueryRetriever` | — | ⬜ |
| 17 | Reranking | Letting the LLM reorder results | `ContextualCompressionRetriever` + `LLMListwiseRerank` | — | ⬜ |
| 18 | Metadata filtering | "Only TechCrunch articles from October" | `SelfQueryRetriever` | — | ⬜ |

## Part D: Conversations

| # | Step | You learn | LangChain piece | Package | Status |
|---|---|---|---|---|---|
| 19 | Chat history | Passing previous turns to the model | `MessagesPlaceholder`, `InMemoryChatMessageHistory` | — | ⬜ |
| 20 | Follow-up questions | Rewriting "what about him?" into a full question | History-aware rewrite chain | — | ⬜ |
| 21 | Saved history | Keeping chats across restarts | `PostgresChatMessageHistory` | — | ⬜ |

## Part E: Safety and quality

| # | Step | You learn | LangChain piece | Package | Status |
|---|---|---|---|---|---|
| 22 | Input guardrail | Detecting prompt injection / off-topic questions | Structured output + `RunnableBranch` | — | ⬜ |
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
| 10 | `PGVector` is deprecated (since `langchain-postgres` 0.0.14); the replacement is `PGVectorStore`. | Use `PGEngine` + `PGVectorStore` |
| 10 | `init_vectorstore_table` has no "create if missing": it errors if the table exists, and `overwrite_existing=True` drops every stored embedding. | Catch `ProgrammingError` and keep the table |
| 10 | `similarity_search_with_score` means different things per store: `InMemoryVectorStore` returns similarity (higher = better), `PGVectorStore` returns cosine distance (lower = better). | Check each store's score meaning before comparing or thresholding |

## Notes

- This order is a plan, not a contract. Steps can be split, merged or dropped as we go.
- Steps 2–5, 9 and 11–13 need no new packages, because they use what earlier steps installed.
