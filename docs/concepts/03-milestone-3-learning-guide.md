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
