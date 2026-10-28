"""The `rag` command, the single entry point:

    uv run rag ingest              load, split, embed and store the corpus (skips stored chunks)
    uv run rag ask "question"      answer one question
    uv run rag chat                ask follow-up questions in one conversation
"""

import argparse

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from rag_langchain.generation import answer
from rag_langchain.ingestion import ingest


def print_answer(result: dict) -> None:
    print(f"\n{result['answer']}\n\nSources:")
    for n, document in enumerate(result["documents"], start=1):
        print(f"  [{n}] {document.metadata['title']} — {document.metadata['source']}")
        print(f"      {document.metadata['url']}")


def chat() -> None:
    history: list[BaseMessage] = []
    print("Ask about the news. An empty line quits.")
    try:
        while question := input("\n> ").strip():
            result = answer(question, history)
            print_answer(result)
            history += [HumanMessage(question), AIMessage(result["answer"])]
    except (EOFError, KeyboardInterrupt):
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="RAG over the MultiHop-RAG news corpus.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="load, split, embed and store the corpus (skips stored chunks)")
    ask = commands.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    commands.add_parser("chat", help="ask follow-up questions in one conversation")
    args = parser.parse_args()

    if args.command == "ingest":
        ingest()
    elif args.command == "ask":
        print_answer(answer(args.question))
    else:
        chat()
