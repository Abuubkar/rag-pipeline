# Step 20: follow-up questions (rewrite the question before searching)

Notes for [`step20_follow_up.py`](step20_follow_up.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ LLM rewrite ("query condensing") | The LLM reads the history and turns the follow-up into a stand-alone question; that gets searched | The standard fix for chat RAG | One extra LLM call per follow-up; the rewrite can be wrong |
| ✅ Skip the rewrite on the first turn | No history means nothing to resolve | Always | — |
| `create_history_aware_retriever` | The same rewrite, ready-made | — | Moved to legacy `langchain-classic` in v1 |
| Search with history + question pasted together | No LLM: the search text includes the earlier turns | Quick hack | Long, noisy search text; the old topic drowns out the new one |
| Search with both the original and the rewrite | Multi-query (step 16) with the rewrite as one more query | Rewrites you don't fully trust | More chunks to rerank |
| Let an agent write the search | The model calls a search tool with its own query (step 26) | Agents | Less control over when and what it searches |

## This step's code

**When to use**

- Chat over documents, where users say "he", "it", "that one" or "what about TechCrunch?".

**Limitations**

- One extra LLM call on every follow-up, even when the question is already complete.
- The rewrite resolves *who*, but "TechCrunch" is still just words in the search: turn 2 found 3 The Verge chunks,
  1 Fortune and 1 TechCrunch. The answer was right because that one TechCrunch chunk made it in. A source filter
  (step 18's planner, run on the rewritten question) would fix this.
- The rewrite can change the meaning. "What was my first question?" was rewritten into the first question itself,
  so it searched for SBF again. The answer was still right, because the history is in the answer prompt, but it
  was a wasted search. Deciding whether to search at all is routing (step 22).
- Same as step 19: the history is in memory only and grows every turn.
