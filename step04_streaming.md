# Step 4: streaming

Notes for [`step04_streaming.py`](step04_streaming.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| invoke | Wait for the full result | Scripts, structured output, background jobs | The user waits for the whole answer |
| ✅ stream | Get chunks as they are generated | Chat UIs | You can't check the answer before the user sees it |
| batch | Many inputs in parallel | Evaluation, bulk jobs, grading chunks (step 17) | Ollama only runs a few requests at once |
| async (ainvoke, astream) | Non-blocking versions of the same calls | Web servers (FastAPI) serving many users at once | async/await everywhere in your code |
| astream_events | Stream events from every step: retriever, LLM, parser | UIs that show progress ("searching…", then the sources, then the answer) | More event types to handle |

## This step's code

**When to use**

- A person reads the answer (chat UI): the first words appear in about a second.

**Limitations**

- The total time is the same; streaming only changes when you see the first words.
- Steps that need the whole output (parsing JSON, structured output) only finish at the end.
- The user has already seen the answer before you can check it (guardrails, step 23).
