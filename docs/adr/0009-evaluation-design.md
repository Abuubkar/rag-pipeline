# 0009. Evaluation: MultiHop-RAG golden set, string-match answers, runs saved in Postgres

- Status: accepted
- Date: 2026-11-08

## Context
Milestone 3 changes retrieval (hybrid search, chunkers, rewriting, reranking). Without numbers we
can't tell whether a change helps. We need an evaluation that is cheap enough to run often,
needs no paid API, and can be reproduced later.

## Decision
- **Golden set:** the 2,556 MultiHop-RAG questions at the pinned dataset revision. Evidence
  articles are mapped to corpus ids by title (all 5,908 references resolve).
- **Split:** stratified by question type, 20% dev / 80% test. Items are ordered by a hash of
  their id, so an item never changes split. Tune on dev; run test only for final numbers.
- **Retrieval metrics** (no LLM, about 30 s on dev): hit@k, recall@k, MRR at the article level,
  plus evidence-fact recall (the exact evidence sentence appears in the top-k chunks).
  Null queries are skipped because they have no evidence.
- **Answer metrics** (one LLM call per question): the reference answers are short, so the
  first sentence of the answer is normalised and compared. *Accuracy* means the reference
  appears as whole words; *exact match* means equality. On null queries, correct means the
  model refused. Refusal rate and citation rate are reported too.
- **Baseline:** a no-retrieval run with the same model (`closed_book` prompt) measures what the
  model already knows. MultiHop-RAG is late-2023 news, so contamination is possible.
- **Runs are saved** in `eval_runs` / `eval_results` with the config snapshot and hash, index
  version, git commit (plus a dirty flag), prompt label, model and dataset revision.
  Generation uses temperature 0 and a fixed seed.

## Alternatives considered
- **LLM-as-judge (RAGAS) now:** better at paraphrases, but slow and costly on a local model,
  and the judge needs a different model family. Moved to Milestone 4 with Groq.
- **Random seeded split:** it changes when items are added or reordered; the hash split doesn't.
- **Results only in Langfuse:** traces aren't a good place for diffing runs question by
  question. Postgres is already there; Langfuse still gets one trace per run.

## Consequences
- String matching under-counts correct paraphrases ("Bankman-Fried" for "Sam Bankman-Fried")
  and can be fooled by a lead sentence that hedges. Compare runs with each other, not against
  an absolute bar.
- A full dev answer run is slow on a 16 GB laptop, so `--limit` is used while iterating.
- Results are only reproducible when the run's commit isn't dirty and Ollama serves the same
  model weights.
