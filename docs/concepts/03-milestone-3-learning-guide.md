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
