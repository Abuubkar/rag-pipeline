"""Step 21b: saved history the v1 way (a LangGraph checkpointer).

Step 21 saved messages with our own table and two functions (load_history, save_turn). LangGraph
does both for you, but only for a graph, so the step 20 chain runs as the one node of a graph:

    START → answer → END          state = {"messages": [...]}

On every graph.invoke(..., {"configurable": {"thread_id": "demo"}}):
    1. the checkpointer loads the last saved state of thread "demo"
    2. the new HumanMessage is appended to state["messages"] (the add_messages reducer)
    3. the answer node runs and returns {"messages": [AIMessage]}, which is appended too
    4. the checkpointer saves the new state to Postgres

Run each question as a separate program, so nothing is left in memory between them:
    uv run walkthrough/step21b_checkpointer.py demo "Who is facing a criminal trial for fraud, as reported by The Verge?"
    uv run walkthrough/step21b_checkpointer.py demo "What did TechCrunch report about him?"
Inspect a saved thread (no question = only print what is saved):
    uv run walkthrough/step21b_checkpointer.py demo
(Postgres from step 10 and Ollama must be running)
"""

import sys

from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, MessagesState, StateGraph

from step20_follow_up import chain

# psycopg's URL format: step 10's URL without SQLAlchemy's "+psycopg" driver part.
DATABASE_URL = "postgresql://rag:rag@localhost:5434/rag"


def answer(state: MessagesState) -> dict:
    """The only node. state["messages"] = the saved history + the new question (last)."""
    *history, question = state["messages"]
    result = chain.invoke({"question": question.content, "history": history})
    print(f"  searched for: {result['search_query']!r}")
    return {"messages": [AIMessage(result["answer"])]}  # appended to the state, not replacing it


builder = StateGraph(MessagesState)
builder.add_node(answer)  # node name = function name: "answer"
builder.add_edge(START, "answer")  # no edge out of "answer", so the graph ends after it

if __name__ == "__main__":
    thread_id = sys.argv[1]
    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()  # creates LangGraph's tables if they don't exist
        graph = builder.compile(checkpointer=checkpointer)

        if len(sys.argv) > 2:
            question = sys.argv[2]
            print(f"saved before this run: {len(graph.get_state(config).values.get('messages', []))} messages")
            print(f"\nQ: {question}")
            final_state = graph.invoke({"messages": [("human", question)]}, config)
            print(f"A: {final_state['messages'][-1].content}")

        # Debugging: what the checkpointer has saved for this thread.
        print(f"\nthread {thread_id!r} now holds:")
        for message in graph.get_state(config).values.get("messages", []):
            print(f"  {message.type:5}  {' '.join(message.content.split())[:90]}")
        print(f"checkpoints saved for this thread: {len(list(graph.get_state_history(config)))}")
