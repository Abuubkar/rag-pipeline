"""Step 21: saved history (the conversation survives a restart).

In steps 19 and 20 the history was a Python list, so it was gone when the program exited. Saving it
means writing each message to Postgres and reading them back next time. One row per message:

    chat_messages
    id | conversation_id | message (jsonb)
    1  | demo            | {"type": "human", "data": {"content": "Who is facing …", …}}
    2  | demo            | {"type": "ai",    "data": {"content": "Sam Bankman-Fried …", …}}

messages_to_dict / messages_from_dict (langchain-core) turn messages into JSON and back.
LangChain's PostgresChatMessageHistory did this for you, but it builds on the deprecated
BaseChatMessageHistory. The v1 way is a LangGraph checkpointer (the next step).

Run each question as a separate program, so nothing is left in memory between them:
    uv run walkthrough/step21_saved_history.py demo "Who is facing a criminal trial for fraud, as reported by The Verge?"
    uv run walkthrough/step21_saved_history.py demo "What did TechCrunch report about him?"
(Postgres from step 10 and Ollama must be running)
"""

import json
import sys

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, messages_from_dict, messages_to_dict
from sqlalchemy import create_engine, text

from step10_postgres import DATABASE_URL
from step20_follow_up import chain

engine = create_engine(DATABASE_URL)

with engine.begin() as connection:
    connection.execute(
        text(
            "CREATE TABLE IF NOT EXISTS chat_messages ("
            "id serial PRIMARY KEY, conversation_id text NOT NULL, message jsonb NOT NULL)"
        )
    )


def load_history(conversation_id: str) -> list[BaseMessage]:
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT message FROM chat_messages WHERE conversation_id = :id ORDER BY id"), {"id": conversation_id}
        )
        return messages_from_dict([row.message for row in rows])


def save_turn(conversation_id: str, question: str, answer: str) -> None:
    messages = messages_to_dict([HumanMessage(question), AIMessage(answer)])
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO chat_messages (conversation_id, message) VALUES (:id, CAST(:message AS jsonb))"),
            [{"id": conversation_id, "message": json.dumps(message)} for message in messages],
        )


if __name__ == "__main__":
    conversation_id, question = sys.argv[1], sys.argv[2]

    history = load_history(conversation_id)
    print(f"loaded {len(history)} messages for conversation {conversation_id!r}")

    result = chain.invoke({"question": question, "history": history})
    save_turn(conversation_id, question, result["answer"])

    print(f"\nQ: {question}")
    print(f"  searched for: {result['search_query']!r}")
    print(f"A: {result['answer']}")
