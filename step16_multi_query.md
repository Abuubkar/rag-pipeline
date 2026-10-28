# Step 16: multi-query (ask the question several ways)

Notes for [`step16_multi_query.py`](step16_multi_query.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ Multi-query (+ original question) | The LLM writes several queries; results are merged | Questions with several parts | One extra LLM call; only as good as the queries |
| Query decomposition | Split into sub-questions, answer each, then combine | One answer feeds the next ("the CEO of the company that…") | Many sequential LLM calls |
| HyDE (hypothetical document) | The LLM writes a fake answer; that answer is embedded and searched | Short or vague questions | Made-up details can steer the search |
| Step-back prompting | Also search a more general version of the question | Questions that need background | Can drift too broad |
| RAG-Fusion | Multi-query + RRF across the result lists | When ranking across queries matters | Still benefits from reranking |
| Single query rewrite | The LLM cleans up one query | Messy input; chat follow-ups (step 20) | Still one search |

## This step's code

**When to use**

- Questions with several parts: comparisons, "both X and Y", multi-hop questions.

**Limitations**

- One extra LLM call per question before searching (slower).
- Only as good as the queries the LLM writes; it may write fewer than asked.
- Publication names and dates in the query text don't filter anything (step 18).
- function_calling can crash on a misnamed tool; the fallback then searches only the original question.
