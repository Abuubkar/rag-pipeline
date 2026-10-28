# Step 19: chat history (remembering earlier turns)

Notes for [`step19_chat_history.py`](step19_chat_history.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ Plain list + MessagesPlaceholder | Append each question and answer to a list; send the list with every call | Learning; one process; short chats | Lost when the program exits; you manage one list per conversation |
| RunnableWithMessageHistory + ChatMessageHistory classes | LangChain loads and saves the history around a chain for you | — | Deprecated in v1 (removal in 2.0) |
| LangGraph checkpointer (`create_agent` or a graph + InMemorySaver / PostgresSaver) | Messages saved per `thread_id` | Real apps: many users, survives restarts | Needs `langgraph`; the docs' examples are agents |
| ✅ Send the full history | Every earlier turn goes into every prompt | Short conversations | Prompt grows every turn (slower, more tokens, context limit) |
| Trimming (`trim_messages`) | Keep only the last N messages or tokens | Long chats | Forgets early facts |
| Summarising | The LLM condenses older turns into a summary | Very long chats | Extra LLM calls; the summary loses detail |
| Long-term memory | Store facts about the user; retrieve the relevant ones later | Remembering across separate sessions | A second retrieval system to build and keep correct |

## This step's code

**When to use**

- Follow-up questions in one conversation, while the program runs.

**Limitations**

- Retrieval uses only the new question: "What did TechCrunch report about **him**?" searched those literal words, found
  unrelated TechCrunch roundups, and answered "Insufficient information." Rewriting the question with the history
  is step 20.
- Every question triggers a search, even questions about the conversation. "What was my first question?" retrieved
  an excerpt containing the words "first question" (about polar molecules), and the model trusted it over the
  history. Without excerpts it answered correctly. Fix: decide whether to search at all (routing, step 22; or an
  agent, step 26).
- Citations in the history are stale: turn 1's answer says "[5]", but on turn 2 "[5]" is a different excerpt.
- The history grows every turn and is never trimmed; it's lost when the program exits.
