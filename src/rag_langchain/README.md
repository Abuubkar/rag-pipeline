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
| [`prompts.py`](prompts.py) | All four prompts: rewrite, plan, grade, answer | 2, 13, 17, 18, 19, 20 |
| [`loaders.py`](loaders.py) | The dataset as LangChain Documents | 6 |
| [`vectorstore.py`](vectorstore.py) | The Postgres vector store, and the list of publication names | 10 |
| [`ingestion.py`](ingestion.py) | Load → split → embed only the chunks not stored yet | 7, 10 |
| [`retrieval.py`](retrieval.py) | Rewrite follow-ups with the history; plan (query, publication) searches → filtered vector search → dedupe | 16, 18, 20 |
| [`reranking.py`](reranking.py) | The LLM scores each candidate 0–10, seeing the full question and the search that found it; keep the best of every search, then the top scores up to 5 | 17 |
| [`generation.py`](generation.py) | The full chain: rewrite → retrieve → rerank → answer with citations and history | 12, 13, 19, 20 |

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

    gen->>ret: standalone_question(question, history)
    ret->>llm: REWRITE_PROMPT, only when there is history ("him" → "Sam Bankman-Fried")
    ret-->>gen: search_question

    gen->>ret: retrieve(search_question)
    ret->>llm: PLAN_PROMPT → [(query, source), …]
    ret->>pg: one similarity_search per plan,<br/>filter={"source": …} when set,<br/>plus the original question
    pg-->>ret: 5 chunks per search
    ret-->>gen: candidates: chunks, each tagged with metadata["query"]

    gen->>rr: rerank(search_question, candidates)
    rr->>llm: GRADE_PROMPT per chunk: question + its search query (in parallel) → score 0–10
    rr-->>gen: best chunk of every search + top scores (score > 0)

    gen->>llm: ANSWER_PROMPT: numbered excerpts + history + original question
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
| `.assign(search_question=…)` | + `search_question`: the question rewritten to stand alone (the same question on the first turn) |
| `.assign(candidates=…)` | + `candidates`: chunks from every search, deduplicated, each with `metadata["query"]` = the search that found it (often 10–25 chunks) |
| `.assign(documents=…)` | + `documents`: the best chunk of every search, then the highest scores up to 5 |
| `.assign(context=…)` | + `context`: the documents as numbered text |
| `.assign(answer=…)` | + `answer`: the LLM's reply with `[n]` citations |

## Known limits

- Slow: one LLM call to rewrite a follow-up, one to plan, one per candidate to rerank, one to answer (about 2–3 minutes per question
  on an M1 Pro).
- Questions about the conversation itself ("what was my first question?") still trigger a search; deciding
  whether to search at all is routing (walkthrough step 22).
- Chat history lives only in memory while `rag chat` runs.
