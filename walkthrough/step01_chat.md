# Step 1: call a local LLM through LangChain

Notes for [`step01_chat.py`](step01_chat.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ Local model (Ollama) | The model runs on your machine | Privacy matters, no per-call cost, offline work | Slower and weaker than big hosted models; RAM limits model size (16 GB → ~12B) |
| Hosted API (OpenAI, Anthropic, Google) | The provider runs the model; you call an API | You need the best quality, speed or a long context | Cost per token, data leaves your machine, rate limits |
| ✅ temperature=0 | Always picks the most likely next token | RAG, extraction, anything you want to repeat and test | Less varied wording; still not 100% identical across machines or versions |
| Higher temperature (0.7–1) | Samples less likely tokens too | Creative writing, brainstorming | A different answer every run, so harder to test |
| ✅ reasoning=False | Answers directly | Simple tasks where speed matters | Weaker on multi-step logic |
| reasoning=True ("thinking") | The model writes hidden reasoning before answering | Math, planning, multi-step questions | Many extra tokens, so slower |

## This step's code

**When to use**

- One-off calls to a chat model, prototyping a prompt.
- Data must stay on your machine: Ollama runs the model locally.

**Limitations**

- The model only knows its training data, not our news articles. That's what RAG fixes.
- No memory: every invoke() is independent (chat history comes in step 19).
- A local 12B model is slower and weaker than large hosted models.
- The reply is free text, so code can't reliably act on it (step 5).
