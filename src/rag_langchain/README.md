# rag_langchain

A production-style RAG pipeline over the MultiHop-RAG news corpus. It uses only the preferred pieces from
[`LADDER.md`](../../LADDER.md); the step-by-step versions are in [`walkthrough/`](../../walkthrough/).

```bash
uv run rag ingest              # load, split, embed and store the corpus (skips stored chunks)
uv run rag ask "question"      # answer one question, with sources
uv run rag chat                # follow-up questions in one conversation
```

## Modules

The names follow the RAG stages in order: loaders → ingestion → vectorstore → retrieval → reranking → generation.

| Module | Job | Walkthrough steps |
|---|---|---|
| [`cli.py`](cli.py) | The `rag` command: parses arguments, prints answers, runs the chat loop | — |
| [`config.py`](config.py) | Settings from environment variables (`RAG_DATABASE_URL`, `RAG_CHAT_MODEL`, …) | — |
| [`llm.py`](llm.py) | The chat model (gemma4 via Ollama) | 1 |
| [`embeddings.py`](embeddings.py) | nomic-embed-text with its `search_document:` / `search_query:` prefixes | 8, 9 |
| [`prompts.py`](prompts.py) | All three prompts: plan, grade, answer | 2, 13, 17, 18, 19 |
| [`loaders.py`](loaders.py) | The dataset as LangChain Documents | 6 |
| [`vectorstore.py`](vectorstore.py) | The Postgres vector store, and the list of publication names | 10 |
| [`ingestion.py`](ingestion.py) | Load → split → embed only the chunks not stored yet | 7, 10 |
| [`retrieval.py`](retrieval.py) | Plan (query, publication) searches → filtered vector search → dedupe | 16, 18 |
| [`reranking.py`](reranking.py) | The LLM scores each candidate 0–10; keep the top 5 | 17 |
| [`generation.py`](generation.py) | The full chain: retrieve → rerank → answer with citations and history | 12, 13, 19 |

## Who imports whom

Imports only point downwards, and no module does any work when imported: models and connections are
created on first use (`@cache`).

```mermaid
flowchart TD
    cli[cli.py] --> generation[generation.py]
    cli --> ingestion[ingestion.py]

    generation --> retrieval[retrieval.py]
    generation --> reranking[reranking.py]
    generation --> prompts[prompts.py]
    generation --> llm[llm.py]

    retrieval --> vectorstore[vectorstore.py]
    retrieval --> prompts
    retrieval --> llm
    reranking --> prompts
    reranking --> llm

    ingestion --> loaders[loaders.py]
    ingestion --> vectorstore
    vectorstore --> embeddings[embeddings.py]

    llm --> config[config.py]
    embeddings --> config
    vectorstore --> config
    retrieval --> config
    reranking --> config
    ingestion --> config
```

## `rag ingest`: from the dataset to Postgres

```mermaid
flowchart LR
    hf[(Hugging Face<br/>MultiHop-RAG)] -->|loaders.py<br/>609 articles| split[ingestion.py<br/>split into 8,448 chunks]
    split -->|stable id per chunk| check{already stored?}
    check -->|yes| skip[skip]
    check -->|no| embed[embeddings.py<br/>search_document: prefix]
    embed --> pg[(Postgres + pgvector<br/>vectorstore.py)]
```

## `rag ask`: from a question to an answer

```mermaid
sequenceDiagram
    actor User
    participant cli as cli.py
    participant gen as generation.py
    participant ret as retrieval.py
    participant pg as vectorstore.py (Postgres)
    participant rr as reranking.py
    participant llm as llm.py (gemma4)

    User->>cli: rag ask "question"
    cli->>gen: answer(question, history)

    gen->>ret: retrieve(question)
    ret->>llm: PLAN_PROMPT → [(query, source), …]
    ret->>pg: one similarity_search per plan,<br/>filter={"source": …} when set,<br/>plus the original question
    pg-->>ret: 5 chunks per search
    ret-->>gen: candidates (deduplicated)

    gen->>rr: rerank(question, candidates)
    rr->>llm: GRADE_PROMPT per chunk (in parallel) → score 0–10
    rr-->>gen: top 5 chunks (score > 0)

    gen->>llm: ANSWER_PROMPT: numbered excerpts + history + question
    llm-->>gen: answer with [n] citations
    gen-->>cli: {answer, documents, …}
    cli-->>User: answer + sources
```

`rag chat` does the same per question, and passes the earlier questions and answers as `history`.

## The data that flows through `generation.py`

`rag_chain()` is one LCEL chain; each `.assign` adds a key to the same dict (ladder rung 4):

| After | Dict keys |
|---|---|
| input | `question`, `history` |
| `.assign(candidates=…)` | + `candidates`: every unique chunk from the planned searches (often 10–25) |
| `.assign(documents=…)` | + `documents`: the top 5 after reranking |
| `.assign(context=…)` | + `context`: the documents as numbered text |
| `.assign(answer=…)` | + `answer`: the LLM's reply with `[n]` citations |

## Known limits

- Slow: one LLM call to plan, one per candidate to rerank, one to answer (about 2–3 minutes per question
  on an M1 Pro).
- Follow-ups like "what about **him**?" search the literal words; rewriting them with the history is
  walkthrough step 20.
- Chat history lives only in memory while `rag chat` runs.
