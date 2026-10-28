# The LangChain ladder: which piece to use

Like Django (function views → class-based views → generic views), LangChain has rungs. Each rung builds on
the one below. This page says which rung every piece is on, which steps teach it, and whether real apps use it.

## The rungs

| Rung | Django equivalent | LangChain piece | Steps | Use in real apps? |
|---|---|---|---|---|
| 1. Call the model by hand | Function view | `llm.invoke([SystemMessage(...), HumanMessage(...)])` | [1](walkthrough/step01_chat.py) | Rarely: quick scripts only |
| 2. Templates | Templates | `ChatPromptTemplate` with `{placeholders}`, `MessagesPlaceholder` for message lists | [2](walkthrough/step02_prompts.py), [19](walkthrough/step19_chat_history.py) | ✅ Every LLM call |
| 3. One interface: Runnables | Class-based `View` | Anything with `invoke` / `stream` / `batch`, joined with `\|`: `prompt \| llm \| StrOutputParser()` | [3](walkthrough/step03_chains.py), [4](walkthrough/step04_streaming.py), [12](walkthrough/step12_rag_chain.py) | ✅ The base of every pipeline |
| 4. Composing Runnables | Mixins, passing context | `RunnablePassthrough.assign` (carry several values in one dict), plain functions in a chain, `RunnableLambda`, `.map()`, `.with_fallbacks()` | [13](walkthrough/step13_citations.py), [16](walkthrough/step16_multi_query.py), [17](walkthrough/step17_rerank.py), [18](walkthrough/step18_metadata_filter.py) | ✅ The normal v1 way to build a pipeline |
| 5. Ready-made building blocks | Generic views (`ListView`) | `with_structured_output`, `vector_store.as_retriever()`, store features (hybrid search, metadata filters), loaders, splitters, embeddings | [5](walkthrough/step05_structured.py), [6](walkthrough/step06_load.py)–[11](walkthrough/step11_retriever.py), [14](walkthrough/step14_keyword.py), [15](walkthrough/step15_hybrid.py) | ✅ Whenever one fits |
| 6. Stateful flows and agents | A full framework (DRF viewsets) | LangGraph: graphs with state, loops and a checkpointer; `create_agent` with tools | 21, 26 (planned) | ✅ When the flow needs loops, memory across runs or tool use |

Not on the ladder any more (legacy or deprecated in v1, so we skipped them):

| Piece | Status | Use instead |
|---|---|---|
| `EnsembleRetriever`, `MultiQueryRetriever`, `ContextualCompressionRetriever`, `SelfQueryRetriever` | Legacy (`langchain-classic`) | Store features (rung 5) + small chains (rung 4): steps 15–18 |
| `RunnableWithMessageHistory`, `InMemoryChatMessageHistory`, `PostgresChatMessageHistory` | Deprecated (removal in 2.0) | A message list (step 19); a LangGraph checkpointer to persist (rung 6) |
| `PGVector` | Deprecated | `PGEngine` + `PGVectorStore` (step 10) |
| Loaders and retrievers in `langchain-community` (e.g. `BM25Retriever`, `HuggingFaceDatasetLoader`) | Package sunset | Your own `BaseLoader` (step 6), store features (step 14) |

## Use this, not that

| Situation | Use | Not | Why |
|---|---|---|---|
| Any LLM call | `ChatPromptTemplate` | f-strings or hand-built message lists | Reusable, a Runnable, roles built in |
| Get a string back | `prompt \| llm \| StrOutputParser()` | `llm.invoke(...).content` | Works the same with `stream` and `batch` |
| Get data back | `llm.with_structured_output(Model)` | Parsing JSON from text yourself | Validated objects |
| Keep the question, documents and answer together | `RunnablePassthrough.assign(...)` | Nested dicts or calling steps one by one | One dict grows step by step |
| Run one Runnable on every item of a list | `runnable.map()` | A Python `for` loop with `invoke` | Runs in parallel, stays a Runnable |
| A step can fail (bad LLM output) | `.with_fallbacks([...])` | `try/except` around the whole chain | The fallback stays inside the chain |
| Search | `vector_store.as_retriever()` or a function using the store | Hand-written cosine similarity | The store does it, with indexes and filters |
| Keyword + vector search | The store's hybrid search | `EnsembleRetriever` | Legacy package |
| Chat memory in one process | A list + `MessagesPlaceholder` | `RunnableWithMessageHistory` | Deprecated |
| Memory across restarts, loops, tool use | LangGraph (checkpointer, `create_agent`) | Building loops with LCEL | LCEL has no loops or saved state |

## How the production pipeline uses the ladder

[`src/`](src/) holds the production-style pipeline. It uses only the ✅ pieces above, one clear job per module.
