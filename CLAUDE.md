# CLAUDE.md — `langchain-only` branch

## Why this branch exists

The user is **learning AI**. On `main`, the first milestone added too many packages, directories
and abstractions at once, which defeated the purpose of learning. This orphan branch rebuilds the
RAG pipeline **slowly, one small feature at a time, using LangChain only**, so the user can see:

- what each piece of a RAG pipeline is and why it exists, and
- how much LangChain alone can do, and where it falls short.

Those limits are what would later justify LangGraph, LangSmith or Langfuse.

## Rules (most important first)

1. **One small feature per step.** Build the smallest working piece, explain it, and stop. Don't
   implement ahead of the current step.

2. **Install a package only when code imports it.**
   - Never install packages "for later". If no file imports a package, it must not be in
     `pyproject.toml`.
   - Add packages **one at a time**, as the code being written needs them.
   - **Before installing**, show the package name and why this step needs it, and wait for approval.

3. **Create a directory or file only when the current step needs it.**
   - No scaffolding, no empty folders, no placeholder files.
   - The user doesn't yet know what terms like "ingestion" mean, so every new directory gets a
     one-line explanation of what it's for, at the moment it's created.

4. **LangChain only.**
   - Use the LangChain ecosystem (`langchain`, `langchain-core`, `langchain-*` integration
     packages) for the whole pipeline.
   - Prefer what LangChain already provides over custom code or other libraries, so its real
     capabilities are visible.
   - When LangChain can't do something well, **say so explicitly**. That's a learning point, not
     something to work around silently.
   - Don't bring in LangGraph, LlamaIndex, Langfuse or other frameworks unless the user decides to.

5. **Ask before every commit.** Each commit is one small feature. Show what changed and wait for
   the user's approval. Never commit or push on your own.

6. **No tests** unless the user asks for them.

7. **Ask why when the purpose of a request is unclear**, instead of guessing and over-building.

## Style

- Minimal code; readable over clever. No abstractions until there are two real uses.
- Explain each new concept briefly when it first appears (in chat, plus a short comment in code
  where it helps).
- Every step file `stepNN_name.py` gets a matching `stepNN_name.md`. The project is also for
  learning RAG and AI engineering, not just LangChain, so the `.md` has:
  - **Approaches:** a table of this step's approach and its alternatives (how it works, use it
    when, watch out for), with ✅ on the one the code uses.
  - **This step's code:** when to use it, and its limitations.
- Context from `main` (dataset: MultiHop-RAG; local models via Ollama; Postgres + pgvector) can be
  reused, but only introduced when a step needs it.
