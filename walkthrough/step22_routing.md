# Step 22: routing (and an input guardrail)

Notes for [`step22_routing.py`](step22_routing.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ LLM router with structured output | The LLM picks one label from a fixed list (`Literal`); code runs that path | A few clear routes; no training data | One extra LLM call per message; it can pick wrong |
| ✅ `RunnableBranch` | (condition, runnable) pairs plus a default; the first true condition wins | Fixed routes in LCEL | — |
| A function that returns a runnable | `RunnableLambda(lambda x: news_chain if … else …)` | Routes that need more logic than a condition list | Same result; less readable as the routes grow |
| Embedding router | Compare the message's embedding with example messages per route; nearest wins | Fast and cheap, no LLM call | Needs good examples per route; thresholds to tune |
| Keyword rules | `if "ignore your instructions" in text` | Obvious cases, as a first filter | Easy to get around; many false matches |
| LangGraph conditional edges | `add_conditional_edges(node, route_fn)` chooses the next node | The app is already a graph (step 21b) | Needs `langgraph` |
| Guardrail models / services (Llama Guard, NeMo Guardrails, provider moderation) | A model trained to flag unsafe or injected input | Public apps; real safety needs | Another model or service; tuned for safety, not topic |
| Let an agent decide (step 26) | The model chooses whether to call the search tool | Open-ended assistants | Less control; harder to test |

## This step's code

**When to use**

- A chat over your documents that also gets questions about the chat, small talk, or off-topic requests.

**Limitations**

- One more LLM call per message (the router), on top of rewrite + answer.
- A wrong route is silent: a news question routed to "other" gets refused, and one routed to "conversation" is
  answered without sources. Log the route (the script prints it) and test the router on its own.
- It is not real security. "Ignore your previous instructions" was caught here, but an injection hidden inside a
  news-looking question goes to the news route. Prompt injection can also arrive inside retrieved documents,
  which this router never sees.
- The refusal is fixed text, so it can't say what it *can* help with for that request.
- Three LLM prompts now see the history (router, rewrite, answer), so long chats cost three times more.
