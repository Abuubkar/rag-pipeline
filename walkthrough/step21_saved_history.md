# Step 21: saved history (the conversation survives a restart)

Notes for [`step21_saved_history.py`](step21_saved_history.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ Own table, one row per message | INSERT each question and answer; SELECT them in order next time | You want to see and own the storage | You write load/save, cleanup and indexes yourself |
| Own table, one row per conversation | The whole message list as one JSON value, overwritten each turn | Short chats; simplest reads | Rewrites the whole list every turn; two writers can overwrite each other |
| `PostgresChatMessageHistory` (langchain-postgres) | A ready-made class with the same table shape | — | Builds on the deprecated `BaseChatMessageHistory` (removal in 2.0) |
| LangGraph checkpointer (`PostgresSaver`) | Saves the whole graph state per `thread_id`, after every step | Real apps; the v1 way; also resumes interrupted runs | Needs `langgraph` + `langgraph-checkpoint-postgres`; the chain must run inside a graph |
| Redis / a key-value store | The history kept in a fast cache, often with an expiry | Many users, short-lived chats | Another service to run; lost on expiry |

## This step's code

**When to use**

- A conversation that must survive restarts, or be continued later by `conversation_id`.

**Limitations**

- No users or access control: anyone who knows a `conversation_id` can read and extend it.
- The whole history is loaded and sent every turn; it's never trimmed (see step 19's trimming and summarising).
- Old conversations are never deleted.
- The table is created at import time, and there's no index on `conversation_id` (fine for a demo, slow at scale).
- If the program crashes between the answer and `save_turn`, that turn is lost.
