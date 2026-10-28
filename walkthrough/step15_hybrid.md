# Step 15: hybrid search (vector + keyword in one query)

Notes for [`step15_hybrid.py`](step15_hybrid.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ Reciprocal rank fusion (RRF) | Score = sum of 1 / (60 + rank) over the lists | Scores on different scales; the usual default | Ignores how much better a result is |
| Weighted sum | Normalise each list's scores to 0–1, then weight and add | One search is clearly more trustworthy | Min-max normalisation reacts to outliers; weights need tuning |
| ✅ Store-native hybrid (PGVectorStore) | One call runs both searches and fuses | Both searches live in one database | Shared-config bug: a new config per call |
| EnsembleRetriever (langchain-classic) | Weighted RRF over any retrievers | Mixing retrievers from different stores | Legacy package |

## This step's code

**When to use**

- Queries that mix exact names and meaning; a safe default when you don't know the query types.

**Limitations**

- RRF uses only ranks and trusts both lists equally; neither judges if a chunk answers the question.
- For long questions keyword search returns nothing, so hybrid behaves like plain vector search.
- primary_top_k / secondary_top_k need tuning per dataset.
- Same shared-config bug as step 14: a new HybridSearchConfig per call.
