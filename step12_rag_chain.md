# Step 12: the first full RAG chain

Notes for [`step12_rag_chain.py`](step12_rag_chain.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ 2-step RAG | Always retrieve, then answer | FAQ and documentation bots; predictable cost | One search; can't adapt to the question |
| Agentic RAG | The LLM decides when and what to search (a tool) | Varied, multi-step questions (step 26) | Slower, less predictable, more LLM calls |
| Hybrid RAG | A fixed pipeline plus checks: rewrite, grade, retry | Quality-critical answers | More moving parts (usually LangGraph) |
| ✅ Stuffing | All chunks go into one prompt | A few chunks that fit the context | Long contexts; models miss facts in the middle |
| Map-reduce | Answer from each chunk separately, then combine | Many chunks | Many LLM calls; links between chunks get lost |
| Refine | Improve the answer chunk by chunk | Long single documents | Sequential, so slow |
| Long context, no retrieval | Put whole documents in the prompt | A small corpus and a large-context model | Cost and latency; can't fit 609 articles |

## This step's code

**When to use**

- Simple Q&A over your documents, where one search finds everything needed ("2-step RAG").

**Limitations**

- One search per question: multi-hop questions miss sources (step 16).
- The answer is only as good as the retrieved chunks; the LLM can't fix missing context.
- Only the answer string comes back, without sources (step 13).
