# Milestone 2: what to learn alongside the code

Milestone 2 adds **evaluation**. Every change in Milestone 3 (hybrid search, other chunkers, query
rewriting, reranking) will be judged by these numbers, so this milestone is where "it looks
better" turns into "it is better, by this much, on these questions".

Read in this order: golden set → retrieval metrics → answer metrics → baseline → saved runs.

---

## 1. Golden sets and dev/test splits (≈ 45 min)
**Learn:** a golden set is questions with known answers *and* known evidence. Why you tune on
**dev** and only report **test**: if you keep changing the pipeline until test looks good, you've
fitted the pipeline to those exact questions, and the number stops predicting real use.
**Stratified** splits keep each question type in the same proportion in dev and test.
**Code:** `evaluation/dataset.py` (`load_golden_set`, `sample`),
`packages/documents/multihop_rag.py::load_questions`
**Exercise:** `uv run main.py eval dataset`. Then change `evaluation.dev_fraction` to 0.3 and run it
again. Items that were in dev stay in dev. Why? (Hint: hash order, not a shuffle.)

## 2. Retrieval metrics (≈ 1 h)
**Learn:** hit@k vs recall@k vs MRR, and why multi-hop questions need recall (every article),
not just hit (any article). Article-level vs chunk-level metrics: the right article can still
come back as the wrong chunk, which is what **fact recall** catches.
**Code:** `evaluation/metrics.py` (`score_retrieval`, `summarize`), `evaluation/experiments.py`
**Exercise:** `uv run main.py eval retrieval`. Look at the gap between hit@6 and recall@6, and
between recall@6 and fact_recall. Each gap is a different failure: missing articles vs the right
article with the wrong chunk.

## 3. Answer metrics without an LLM judge (≈ 45 min)
**Learn:** exact match vs "contains" match; normalising answers (case, punctuation, articles,
citations); why scoring only the *first sentence* works when the prompt says "start with the
direct answer"; what string matching misses (paraphrases) and what it wrongly accepts (a lead
sentence that hedges). Null queries: refusing *is* the correct answer, so refusal counts as
right there and wrong everywhere else.
**Code:** `evaluation/metrics.py` (`score_answer`, `_normalize_answer`, `_first_sentence`)
**Exercise:** `uv run main.py eval show <run>`, then look at individual answers:
`docker exec rag-postgres psql -U rag -d rag -c "select scores->>'correct', scores->>'reference', left(scores->>'answer',100) from eval_results where run_id='<run>' limit 20"`.
Find one answer that's marked wrong but is actually right. That's a judge's job (Milestone 4).

## 4. Baselines: is RAG actually helping? (≈ 30 min)
**Learn:** a **no-retrieval (closed-book)** baseline asks the same model with no sources. If it
scores close to RAG, the model already knows the answers from training (*contamination*), and
the benchmark says little about your retrieval. A baseline turns a single number into a
comparison.
**Code:** `pipeline.py::evaluate_answers(retrieval=False)`, `generation/generate.py::closed_book`,
`config/prompts/closed_book.yaml`
**Exercise:** compare the `--no-retrieval` run with the RAG run per question type. Where does
retrieval help most? Where does it *hurt* (more refusals)?

## 5. Reproducible experiments (≈ 45 min) — what makes it "production"
**Learn:** an eval number is worthless if you can't say what produced it. Each run stores a
config snapshot and its hash, the index version, the git commit (and whether the tree was dirty),
the prompt `name@version`, the model and the dataset revision. Temperature 0 plus a fixed seed
make local generation repeatable. JSON vs JSONB in Postgres (JSONB re-sorts keys; JSON keeps
your order).
**Code:** `evaluation/runs.py`, `core/provenance.py`, `db/eval_registry.py`,
`db/migrations/versions/0002_eval_runs.py`
**ADR:** `docs/adr/0009-evaluation-design.md`
**Exercise:** run `eval retrieval` twice and compare with `eval runs`: same config hash, same
numbers. Then change `retrieval.top_k` to 10, run again, and see the hash change.

## 6. Cost and speed of evaluation (≈ 15 min)
**Learn:** retrieval evals are cheap (one embedding + one SQL query per question) and can run on
every change. Answer evals cost one LLM call per question; on a 16 GB laptop most of that time is
the model *reading* the ~3,000-token context, not writing the answer. That's why `--limit` takes a
fixed, type-balanced sample, and why Milestone 4 judges only a subset with Groq. On a laptop,
also keep **one model in memory at a time**: RAG answer evals first retrieve for every question
(embedding model), unload it, then generate every answer (chat model only).
**Code:** `pipeline.py::evaluate_answers` (the two phases), `packages/embeddings/ollama.py::unload`
**Exercise:** time `eval answers --limit 10` with `retrieval.top_k: 3` vs `6`. The prompt is half
the size: how much faster is it, and what happens to accuracy?

---

## Milestone 2 results (dev split)

Index `sentence-25bd5b8fef` (LlamaIndex sentence splitter, 512/64 tokens, nomic-embed-text),
generator `gemma4:12b`, commit `085a59e`. See them with `uv run main.py eval runs`.

**Retrieval, dense search** (all 451 answerable dev questions; two runs gave identical scores):

| metric | comparison | inference | temporal | all |
|---|---|---|---|---|
| hit@6 | 0.89 | 0.85 | 0.94 | 0.89 |
| recall@6 | 0.62 | 0.44 | 0.65 | 0.56 |
| recall@10 | 0.70 | 0.56 | 0.73 | 0.66 |
| MRR | 0.74 | 0.68 | 0.78 | 0.73 |
| fact recall @6 | 0.53 | 0.30 | 0.51 | 0.44 |

**Answers, RAG vs no retrieval** (100-question type-balanced dev sample):

| accuracy | comparison | inference | null | temporal | all |
|---|---|---|---|---|---|
| no retrieval | 0.04 | **0.71** | 1.00 | 0.12 | 0.42 |
| RAG | **0.50** | 0.57 | 1.00 | **0.25** | **0.53** |
| RAG refusal rate | 0.32 | 0.40 | 1.00 | 0.42 | 0.46 |

**What the numbers say**

- Search almost always finds *one* gold article (hit@6 0.89) but only about half of them
  (recall@6 0.56). Multi-hop questions need all of them, so the model often refuses: it refuses
  46% of all questions, though only 13% are null queries.
- Retrieval helps a lot where the model can't know the answer (comparison 0.04 → 0.50,
  temporal 0.12 → 0.25).
- On inference questions RAG is *worse* than no retrieval (0.71 → 0.57). The model already knows
  many of them from training (e.g. "Sam Bankman-Fried"), and with incomplete sources the prompt
  makes it refuse instead. This is contamination, and it's why the baseline exists.
- Exact match (0.17) is far below accuracy (0.53): answers are right but phrased as sentences.
  String matching is a floor; Milestone 4's judge will score paraphrases.
- **Milestone 3's target:** raise recall@6 and fact recall (hybrid search, better chunks,
  multi-query, reranking), then check that RAG refusals drop and accuracy rises.

