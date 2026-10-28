# Step 11: retrievers

Notes for [`step11_retriever.py`](step11_retriever.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ similarity | The k closest chunks | The default for most questions | Often several chunks of one article |
| ✅ mmr (fetch_k, lambda_mult) | Picks close AND different chunks; lambda_mult 1 = only relevance, 0 = only variety | Answers that need several sources | Variety can pull in off-topic chunks |
| similarity_score_threshold | Only chunks above a score | "Nothing relevant" must be a possible result | Each store and model has its own score scale; needs tuning |
| Choosing k | How many chunks reach the prompt | Small k: focused; large k: better recall | Large k adds noise and makes prompts longer |

## This step's code

**When to use**

- Whenever retrieval goes inside a chain: a retriever is a Runnable.
- MMR: when the top results are near-duplicates from one article.

**Limitations**

- k is fixed: every question gets 5 chunks, whether it needs 1 or 10.
- Similarity often returns several chunks of the same article.
- MMR's variety can pull in unrelated articles (ChatGPT and FTX for the FTC question).
