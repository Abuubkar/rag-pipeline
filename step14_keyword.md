# Step 14: keyword search (Postgres full-text search)

Notes for [`step14_keyword.py`](step14_keyword.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ Postgres full-text search | tsvector / tsquery, ranked by ts_rank_cd | Already on Postgres; no extra service | Not BM25: ignores how rare a word is; every word required (plainto_tsquery) |
| websearch_to_tsquery / OR queries | Google-style syntax, any-word matching | Long natural questions | PGVectorStore hard-codes plainto_tsquery; needs your own SQL |
| BM25 (rank-bm25, BM25Retriever) | Ranks by word frequency, word rarity (IDF) and document length | Small corpora held in memory | In sunset langchain-community; rebuilt every run |
| Elasticsearch / OpenSearch | BM25 at scale, language analyzers | Large text-search systems | A separate cluster to run |
| Learned sparse (SPLADE) | A model predicts weighted words, including related ones | Keyword precision with some synonym matching | Needs the model and a sparse index |

## This step's code

**When to use**

- Short queries with exact names, product names, codes or rare words.

**Limitations**

- plainto_tsquery requires every word, so long natural questions get 0 results.
- Ranks by how often the words appear, not by meaning.
- A shared HybridSearchConfig keeps reusing the first query: create a new one per call.
- No keyword-only mode: the query is still embedded, though the vector half is unused.
