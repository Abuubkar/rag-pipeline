# Milestone 3: what to learn alongside the code

Milestone 3 improves **retrieval**, one change at a time, and judges each change with the
Milestone 2 eval. Every experiment is a labelled run in Postgres, and `docs/results/` shows the
before/after tables and charts. Sections are added as each step lands.

---

## 1. Keyword search and hybrid fusion (step 3.1, ≈ 1.5 h)
**Learn:**
- How Postgres full-text search works: `to_tsvector` (stemmed terms, stop words dropped),
  `tsquery`, and `ts_rank_cd` (ranks by how close the matched terms are to each other).
- Why a question needs **OR** semantics. `plainto_tsquery` ANDs every word, so a long question
  matches almost nothing.
- Why dense and keyword search fail differently. Dense search misses exact names and numbers;
  keyword search misses paraphrases.
- **Reciprocal Rank Fusion:** `score = Σ 1/(60 + rank)`. It uses ranks only, so the two
  incomparable score scales never need normalising.

**Code:** `packages/vectorstore/pgvector.py::keyword_search`, and `retrieval/search.py` (`rrf_fuse`, `search`).

**Exercise:** `RAG_RETRIEVAL__MODE=keyword uv run main.py ask "..."` against `dense` and `hybrid`.
Find a question where keyword search wins (hint: a rare proper noun).

**Result:** keyword alone is worse than dense (recall@6 0.48 vs 0.56). Hybrid beats both
(0.61), because each search finds articles the other misses.

## 2. Chunking strategies (step 3.2, ≈ 1.5 h)
**Learn:**
- The three chunkers, all at 512 tokens with 64 overlap (counted with one tokenizer, `cl100k_base`):

  | Chunker | How it cuts |
  |---|---|
  | **Sentence** (LlamaIndex) | packs whole sentences up to the size limit |
  | **Recursive** (LangChain) | tries paragraph → line → word separators, and only goes finer when a piece is too long |
  | **Semantic** (LlamaIndex) | embeds each sentence with its neighbours and cuts where similarity drops below the 95th-percentile breakpoint, i.e. at topic shifts |

- Semantic chunks have no size limit (17 to 1,700 tokens here), so chunks over 512 tokens are
  re-split; otherwise the embedder would silently truncate them.
- **Equal budget:** a chunker can "win" just by returning more text. The *context tokens*
  column shows what the LLM would read for each experiment.

**Code:** `packages/chunking/llamaindex.py` (`split_sentence`, `split_semantic`, `_Embedding`),
`packages/chunking/langchain.py::split_recursive`, and `ingestion/chunking.py`.

**Exercise:** `RAG_CHUNKING__SPLITTER=semantic RAG_CHUNKING__BREAKPOINT_PERCENTILE=90 uv run main.py ingest --limit 50 --no-activate`.
Then compare chunk counts with `index list`. A lower percentile gives more, smaller chunks. Why?

**Result:** a tie. Under hybrid search all three reach recall@6 0.61, with fact recall
0.47–0.48 and MRR 0.76–0.78. Semantic does it with the least text (2,668 context tokens vs
2,865 for sentence), but indexing took 5× as long
(17 min vs 3 min) because it embeds every sentence. News articles are short and
on one topic, so where you cut them matters little. That is why sentence chunking stays the
default (ADR 0010). Whether the small differences are real gets checked with paired confidence
intervals in step 3.7.

## 3. Query transformation: rewrite, multi-query, HyDE (step 3.3, ≈ 1.5 h)
**Learn:**
- Three ways to let the LLM change *what gets searched* before searching:

  | Transform | What is searched | LLM output |
  |---|---|---|
  | **Rewrite** | one short keyword query *instead of* the question | 1 line |
  | **Multi-query** | the question + 3 sub-queries; all result lists fused with RRF | 3 lines |
  | **HyDE** | the question + a fake answer passage, embedded **as a document** | ~80 words |

- Why HyDE embeds the passage with the *document* prefix: the passage looks like an article, so
  comparing it document-to-document is closer than question-to-document.
- **Two-phase eval** again: gemma writes every question's queries, is unloaded
  (`OllamaChat.unload`), and only then does nomic search. Two models are never in memory together.
- Cost: one LLM call per question *before* search. Here that was about 3 s for rewrite and
  multi-query, and about 8 s for HyDE: seconds of latency added to every `ask`.

**Code:** `retrieval/query_transform.py` (`transform`, `transform_node`), `retrieval/search.py::search`
(every query and passage becomes one ranked list), `config/prompts/{rewrite,multi_query,hyde}.yaml`,
and `pipeline.py::_transform_all`.

**Debug:** each question's generated queries are saved with its score:
`docker exec rag-postgres psql -U rag -d rag -c "select scores->'searched' from eval_results where run_id='<run>' limit 5"`.

**Result:** none of them is a clear win.

| Transform | recall@6 | fact recall | questions better / worse than hybrid |
|---|--:|--:|--:|
| rewrite | 0.57 | 0.44 | 36 / 83 |
| multi-query | 0.57 | 0.43 | 50 / 80 |
| HyDE | 0.61 | 0.49 | 50 / 41 |

Plain hybrid scores recall@6 0.61 and fact recall 0.48. Why the first two hurt:
- MultiHop-RAG questions are already long and specific. A rewrite throws detail away.
- The multi-query sub-queries lean on publication names and dates ("TechCrunch … October 7
  2023"), which are *metadata*, not chunk text. Step 3.5 (metadata filters) targets exactly this.
- With multi-query, the question's own dense and keyword lists are only 2 of the 8 fused lists,
  so weaker sub-queries outvote it.

HyDE helps a little on inference questions, at 8 s per query. `query_transform` stays `none`.

**Exercise:** change `multi_query.yaml` so the sub-queries leave out publication names and
dates, then re-run with `--limit 100`. Does it close the gap?

## 4. Listwise LLM rerank (step 3.4, ≈ 1.5 h)
**Learn:**
- **Two stages.** Search is fast but shallow: nomic embeds the question and each chunk
  *separately* (a bi-encoder), so it never reads them together. A reranker reads the question
  *with* each candidate, which is slower and sharper. So search fetches 20 candidates and the reranker
  reorders them; the top 6 go to the LLM.
- **Listwise (RankGPT-style):** gemma sees the question and all 20 passages at once (title,
  source, date, text trimmed to 600 characters) and replies with an ordering such as `4 > 1 > 7 > …`.
  Pointwise rerankers instead score each passage alone.
- **Parse defensively.** LLM output is untrusted: invalid or repeated numbers are skipped, and
  candidates it forgets are appended in their search order. A bad reply then degrades to plain
  hybrid instead of crashing.
- **The ceiling:** rerank only reorders. If no gold article is in the top 20, nothing changes, so
  recall@20 of the search stage caps what rerank can achieve.
- **Three-phase eval:** search every question with nomic → unload → rerank every question with gemma
  → unload. Again, only one model is in memory at a time.

**Code:** `retrieval/rerank.py` (`rerank`, `_trim`, `rerank_node`), `config/prompts/rerank.yaml`,
and `pipeline.py::_rerank_all`. When rerank is on, `_search_kwargs` raises `top_k` to `rerank_candidates`.

**Debug:** run one question both ways and compare the sources:
`RAG_RETRIEVAL__RERANK=listwise uv run main.py ask "..."` against the same command without the variable.

**Result:** the biggest gain in Milestone 3 so far. It was measured on the 87-question sample, because
it costs **≈ 51 s per question** (75 min for the sample):

| Run (87 questions) | recall@6 | fact recall | MRR | hit@1 |
|---|--:|--:|--:|--:|
| hybrid | 0.55 | 0.43 | 0.77 | 0.68 |
| hybrid + listwise rerank | **0.69** | **0.53** | **0.89** | **0.84** |

Per question: 33 better, 3 worse, 51 the same. Inference questions gain the most (recall@6
0.41 → 0.58), where the answer depends on reading the passage, not on matching words. Context
tokens stay about the same (≈ 2,800), so the gain is better ordering, not more text.

The trade-off is latency: 51 s added to every `ask` on a laptop is too slow for interactive use.
`rerank` therefore stays `none` by default. It is on for offline evals and for answer quality
experiments. A small cross-encoder (e.g. `bge-reranker`) is the usual production compromise and
stays in the backlog.

**Exercise:** set `RAG_RETRIEVAL__RERANK_CANDIDATES=10` and re-run with `--limit 100`. How much
recall do you lose, and how much time do you save?

## 5. Metadata filters (step 3.5, ≈ 1 h)
**Learn:**
- **Text vs metadata.** "As reported by TechCrunch and The Verge" names *publications*. They live
  in `documents.source`, not in the chunk text, so neither embeddings nor full-text search can
  use them. A SQL `WHERE source IN (...)` can. That is why multi-query hurt in step 3.3.
- **Pre-filtering:** the WHERE runs *inside* the search, so the top 30 are the best chunks
  *from those sources*. Filtering after search (post-filtering) can leave almost nothing.
  For HNSW, `hnsw.iterative_scan` keeps scanning until enough rows pass the filter.
- **How to get the filter from the question** (abstraction ladder):

  | Rung | How | When real apps use it |
  |---|---|---|
  | UI facets | the user picks a source or date in a dropdown | most search products |
  | **Rule-based match** (this step) | match a known, fixed list of values in the text | small, closed vocabularies |
  | LLM self-query | the LLM writes a structured filter (LangChain `SelfQueryRetriever`) | open fields: dates, authors, ranges |

- **Fail open:** if no source is named, no filter is applied. A hard filter that matches
  wrongly removes the right answer, and a filter that matches nothing returns no candidates.

**Code:** `retrieval/metadata_filter.py` (`match_sources`, `sources_for`), `pgvector.py::_source_filter`,
and the `filter_mode` argument of `retrieval/search.py::search`.

**Data check first:** 446 of 451 dev questions name at least one source exactly. In 33 of them,
one gold article comes from a source the question does *not* name, so a hard filter caps them.

**Result:** free (no LLM call, a WHERE clause) and the second-biggest gain so far:

| Run | recall@6 | fact recall | MRR | per question vs hybrid |
|---|--:|--:|--:|--:|
| hybrid (451) | 0.61 | 0.48 | 0.76 | |
| + source filter (451) | **0.66** | **0.55** | **0.83** | 63 better / 14 worse |
| hybrid + listwise rerank (87) | 0.69 | 0.53 | 0.89 | |
| + source filter (87) | 0.59 | 0.49 | 0.84 | |

On the 87-question sample the filter alone (0.59) is still below rerank (0.69), but it costs
milliseconds instead of 51 s. The 14 worse questions are mostly the "unnamed source" cases above.

**Caveat:** MultiHop-RAG questions name their sources because they were generated from article
metadata. Real users rarely do, so this gain is partly a benchmark artefact. The technique
(pre-filtering on metadata) transfers; the size of the gain may not.

**Exercise:** dates. 97 questions name a month ("published on October 7, 2023"). Add a
`published_at` filter (± 1 day) and measure it. Do temporal questions gain?

## 6. Context compression (step 3.6, ≈ 1 h)
**Learn:**
- **Why compress?** Every retrieved token is paid for in prompt length, latency and LLM
  attention. Models also use facts in the middle of a long context less well ("lost in the
  middle"). Compression keeps the same chunks but sends less text.
- **Where it sits:** after search and rerank, on the top_k chunks only. It changes *text*, not
  *which articles*, so recall, hit and MRR stay the same. Fact recall and context tokens move.
- **The ladder:**

  | Rung | How | Cost per question |
  |---|---|---|
  | Fewer chunks (`top_k`) | drop whole chunks | free |
  | **Embedding sentence filter** (this step) | keep the sentences most similar to the question | 1 batch embed call |
  | LLM extractor (LangChain `LLMChainExtractor`) | the LLM copies out the relevant sentences | 1 LLM call per chunk |
  | Token-level (LLMLingua) | a small LM deletes low-information tokens | a second model |

- **Global, not per chunk:** the kept share is chosen across all top_k chunks, so an off-topic chunk
  shrinks a lot and a focused one keeps most of its text. Each chunk keeps at least one sentence,
  so the citation numbers `[n]` don't change.

**Code:** `retrieval/compress.py` (`compress`, `compress_node`), `pipeline.py::_compress_all`
(a fourth eval phase, with nomic only), and `packages/chunking/langchain.py::count_tokens`.

**Result** (451 questions, about 1.7 s per question):

| Run | fact recall | context tokens | facts per 1k tokens |
|---|--:|--:|--:|
| hybrid | 0.48 | 2,865 | 0.17 |
| keep 0.7 | 0.47 | 2,360 (−18%) | 0.20 |
| keep 0.5 | 0.45 | 1,810 (−37%) | 0.25 |

Halving the sentences cuts a third of the tokens but loses some facts: about 6% of the facts
retrieval had found. A fact sentence often doesn't look like the question. In multi-hop
questions, the question mentions both articles, but each fact sentence talks about only one.

Whether the smaller context gives *better answers* is something retrieval metrics can't tell.
That needs the answer eval (the 100-question run planned for later), which can compare
answer accuracy at 1,810 vs 2,865 tokens. Until then, `compression` stays `none`.

**Exercise:** try `RAG_RETRIEVAL__COMPRESSION_KEEP=0.3`. At what point does fact recall drop
sharply? Then compress with the question's *sub-queries* (multi-query) instead of the question.
Do more fact sentences survive?

## 7. Is the difference real? Paired confidence intervals (step 3.7, ≈ 1 h)
**Learn:**
- **Noise.** 451 questions are a sample. Run the same comparison on another 451 and a +0.01 gap
  can flip sign. A confidence interval says how big the gap could plausibly be.
- **Paired, not unpaired.** Both runs answered the *same* questions, so subtract per question
  first. That removes "this question is just hard" (the biggest source of spread), so much
  smaller real gaps become visible.
- **Bootstrap:** resample the 451 per-question differences with replacement, take the mean, and
  repeat 5,000 times. The 2.5th to 97.5th percentile of those means is the 95% CI. No normality
  assumption and no formula, and it works for any metric (recall, MRR, fact recall).
- **Reading it:** if the CI excludes 0, the change is real *for this corpus and question style*.
  With 30 comparisons, about 1 or 2 will exclude 0 by luck, so a CI that only just excludes 0 is weak evidence.
- **Reference = current default** (plain hybrid on the active index), not the M2 baseline.
  Every Milestone 3 decision is "does this beat what we ship now?".

**Code:** `evaluation/compare.py` (`paired_bootstrap`, `compare`), the `forest` chart in `evaluation/charts.py`,
and `report.py::_ci_group`. Per-question scores come from `eval_results` (`runs.results`).

**Result** (vs hybrid; the full tables and forest charts are in `docs/results/`):

| Change | recall@6 Δ [95% CI] | Verdict |
|---|--:|---|
| dense instead of hybrid | −0.045 [−0.068, −0.022] | hybrid is really better |
| semantic / recursive chunks | +0.005 / +0.001 (CIs span 0) | a real tie; MRR +0.02 for semantic only just excludes 0 |
| rewrite / multi-query | −0.041 / −0.032 | really worse |
| HyDE | +0.004 [−0.011, +0.020] | no effect |
| **source filter** | **+0.054 [+0.037, +0.072]** | real, and free |
| **listwise rerank** (87) | **+0.134 [+0.088, +0.180]** | real and large, but 51 s per question |
| compression keep 0.5 | fact recall −0.037 [−0.048, −0.026] | a real loss of facts; answer impact unknown |

The decisions are recorded in ADR 0011.

**Exercise:** compare two runs of the *same* config (run `eval retrieval` twice). Every CI
should contain 0, and the deltas should be exactly 0. Why exactly? (Hint: what in retrieval is random?)

## Answer check: does better retrieval give better answers?
The same 100-question answer eval as Milestone 2, now with hybrid search (the new default):

| Question type | accuracy, dense (M2) | accuracy, hybrid | refusal rate, dense → hybrid |
|---|--:|--:|--:|
| comparison | 0.50 | 0.39 | 0.32 → 0.39 |
| inference | 0.57 | **0.74** | 0.40 → 0.26 |
| temporal | 0.25 | 0.29 | 0.42 → 0.46 |
| null (should refuse) | 1.00 | 1.00 | 1.00 → 1.00 |
| **all** | **0.53** | **0.57** | 0.46 → 0.44 |

Paired: +0.04, 95% CI [−0.04, +0.12], with 11 questions better and 7 worse. That is **not significant on 100
questions**. Better retrieval clearly helps inference questions. Comparison and temporal questions
are limited by the *generator*: gemma still says "Insufficient information" on about 40% of the
answerable ones, even when the evidence was retrieved. That is the target for the next
milestone (prompting and generation), not for more retrieval tuning.
