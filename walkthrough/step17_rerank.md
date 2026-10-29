# Step 17: reranking (let the LLM judge each chunk)

Notes for [`step17_rerank.py`](step17_rerank.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ LLM pointwise | The LLM scores each chunk on its own | No extra model; custom criteria in the prompt | One call per chunk (slow); coarse scores and ties |
| LLM listwise | All chunks in one prompt; the LLM returns an order | Fewer calls; can balance sources | Long prompt; position bias; parsing an order |
| Cross-encoder (bge-reranker, ms-marco MiniLM) | A small model reads question + chunk together and outputs a score | The standard choice: fast and accurate | Another model to run; LangChain's wrapper is in sunset langchain-community |
| Hosted rerank API (Cohere, Voyage) | An API reranks for you | Best quality with little code | Cost; data leaves your machine |
| No reranking | Use the search order as it is | Small k and good retrieval | Noise reaches the LLM's prompt |

## This step's code

**When to use**

- Retrieval returns many chunks with noise, and only the best few should reach the LLM's prompt.

**Limitations**

- One LLM call per chunk: about 1.5 minutes for 2 questions.
- Scores are coarse (only 0, 5 and 10 were used), so ties are common and broken by list order.
- Each chunk is judged alone, so nothing keeps sources balanced (The Verge AND TechCrunch).
- Chunks are graded against the **whole** question. For a two-part question, a chunk answering only one part
  can score 0: the Davis Cup question's rugby fact ("taken their game to new heights") scored 0 and was dropped;
  graded against just the rugby half, the same chunk scored 10. Found later while building `src/`, which
  shows the grader both the full question and the search query that found the chunk, and keeps the best
  chunk of every search.
