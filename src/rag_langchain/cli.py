"""The `rag` command, the single entry point:

    uv run rag ingest              load, split, embed and store the corpus (skips stored chunks)
    uv run rag ask "question"      answer one question
    uv run rag chat                start a saved conversation (prints its thread id)
    uv run rag chat --thread ID    continue a saved conversation
"""

import argparse
import uuid

from rag_langchain.chat import chat_graph
from rag_langchain.generation import answer
from rag_langchain.ingestion import ingest


def print_answer(result: dict) -> None:
    print(f"\n{result['answer']}")
    if result["documents"]:
        print("\nSources:")
    for n, document in enumerate(result["documents"], start=1):
        print(f"  [{n}] {document.metadata['title']} — {document.metadata['source']}")
        print(f"      {document.metadata['url']}")


def chat(thread_id: str | None) -> None:
    thread_id = thread_id or uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}  # which saved conversation to load and extend
    with chat_graph() as graph:
        saved = graph.get_state(config).values.get("messages", [])
        print(f"Conversation {thread_id} ({len(saved)} saved messages). Continue it later with: rag chat --thread {thread_id}")
        print("Ask about the news. An empty line quits.")
        try:
            while question := input("\n> ").strip():
                state = graph.invoke({"messages": [("human", question)]}, config)
                print_answer({"answer": state["messages"][-1].content, "documents": state["documents"]})
        except (EOFError, KeyboardInterrupt):
            print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="RAG over the MultiHop-RAG news corpus.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="load, split, embed and store the corpus (skips stored chunks)")
    ask = commands.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    chat_command = commands.add_parser("chat", help="ask follow-up questions in a saved conversation")
    chat_command.add_argument("--thread", help="the id of a conversation to continue")
    args = parser.parse_args()

    if args.command == "ingest":
        ingest()
    elif args.command == "ask":
        print_answer(answer(args.question))
    else:
        chat(args.thread)
