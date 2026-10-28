# Step 3: chains with LCEL (LangChain Expression Language)

Notes for [`step03_chains.py`](step03_chains.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ LCEL (`prompt \| llm \| parser`) | Runnables joined with `\|` | Straight pipelines; you get invoke / stream / batch for free | No loops; hidden intermediate values; long tracebacks |
| Plain Python functions | Call llm.invoke() in your own code | Small scripts where you want full control | Streaming, batching and tracing must be built by hand |
| LangGraph | A graph of steps with shared state, branches and loops | Agents, retries, "grade and try again", human approval | More concepts and more code |

## This step's code

**When to use**

- A fixed sequence of steps where each output feeds the next (prompt → model → parser).
- You want invoke / stream / batch on the whole pipeline for free.

**Limitations**

- Straight lines only: no loops like "retry until the answer is good" (that's LangGraph).
- Intermediate values (the filled prompt, the raw AIMessage) are hidden unless you print them (step 27).
- An error deep inside a chain gives a long traceback through LangChain internals.
