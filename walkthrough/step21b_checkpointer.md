# Step 21b: saved history the v1 way (a LangGraph checkpointer)

Notes for [`step21b_checkpointer.py`](step21b_checkpointer.py). The approaches table is in
[`step21_saved_history.md`](step21_saved_history.md); this step is its "LangGraph checkpointer" row.

## The LangGraph pieces used

| Piece | What it does |
|---|---|
| `MessagesState` | The state type `{"messages": list}`. Its reducer `add_messages` **appends** whatever a node returns |
| `StateGraph(MessagesState)` | A builder: add nodes and edges to it |
| `add_node(answer)` | A node is a plain function: state in, changed keys out. Named after the function |
| `add_edge(START, "answer")` | Where the graph starts. With no edge out of `answer`, it ends there |
| `compile(checkpointer=…)` | Makes the runnable graph; the checkpointer saves state after every step |
| `PostgresSaver.from_conn_string(url)` | The Postgres checkpointer; used with `with` so its connection is closed. URL in psycopg format (no `+psycopg`) |
| `checkpointer.setup()` | Creates its tables; safe to call every run |
| `{"configurable": {"thread_id": …}}` | Which conversation: same id → same saved state |

## Why 3 checkpoints per question

`get_state_history` after 3 questions showed 9 checkpoints, 3 per run:

| step | source | messages | next | What just happened |
|---|---|---|---|---|
| -1 | input | 0 | `__start__` | The input arrived; not merged yet |
| 0 | loop | 1 | `answer` | `__start__` merged the question into `messages` |
| 1 | loop | 2 | (none) | `answer` ran; its AIMessage was appended. Done |

You can restart from any of these by passing its `checkpoint_id` in the config ("time travel"). That's
something our own step 21 table can't do.

## How to debug it

| To see | Use |
|---|---|
| The saved conversation | `graph.get_state(config).values["messages"]` (the script prints it every run) |
| Every checkpoint | `graph.get_state_history(config)`: `.metadata["step"]`, `.metadata["source"]`, `.next`, `.values` |
| What each node returns while running | `graph.stream(input, config, stream_mode="updates")` |
| The graph's shape | `graph.get_graph().draw_mermaid()` |
| The raw storage | Tables `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`. Values are msgpack bytes, not readable JSON like step 21's table |

## This step's code

**When to use**

- Real apps: the docs' supported way to keep conversations, per `thread_id`.
- The same graph later grows into routing (step 22) or an agent (step 26), and the saving comes for free.

**Limitations**

- Two new packages, and a graph around a chain that is still one straight line.
- The stored state is binary (msgpack). To read a conversation you go through LangGraph, not SQL.
- 3 checkpoints per question, kept forever. Old checkpoints need cleaning up yourself (e.g. delete a thread's rows).
- The full history is still sent to the LLM every turn (no trimming), as in steps 19–21.
- `answer` takes the question from the last message's `.content`; if that is not plain text (images, tool calls), it breaks.
