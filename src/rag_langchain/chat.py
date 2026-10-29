"""Chat with memory: route each message, answer it, and save the conversation per thread. (walkthrough steps 21b, 22)

    START ──route()──┬─ "news"         → news          rag_chain(): rewrite → retrieve → rerank → answer
                     ├─ "conversation" → conversation  answer from the history only, no search
                     └─ "other"        → refuse        fixed text

A LangGraph graph, so a checkpointer can save its state in Postgres after every step, per thread_id.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from rag_langchain.config import settings
from rag_langchain.generation import rag_chain
from rag_langchain.llm import chat_model
from rag_langchain.prompts import CONVERSATION_PROMPT, REFUSAL, ROUTE_PROMPT


class Route(BaseModel):
    route: Literal["news", "conversation", "other"]


class ChatState(MessagesState):
    # "messages" (from MessagesState) is appended to every turn; "documents" is replaced every turn.
    documents: list[Document]  # sources of the latest answer; empty when nothing was searched


@cache
def router() -> Runnable:
    return ROUTE_PROMPT | chat_model().with_structured_output(Route, method="json_schema")


def question_and_history(state: ChatState) -> dict:
    *history, question = state["messages"]
    return {"question": question.content, "history": history}


def route(state: ChatState) -> str:
    """Not a node: called on the way out of START; the name it returns picks the next node."""
    return router().invoke(question_and_history(state)).route


def news(state: ChatState) -> dict:
    result = rag_chain().invoke(question_and_history(state))
    return {"messages": [AIMessage(result["answer"])], "documents": result["documents"]}


def conversation(state: ChatState) -> dict:
    reply = (CONVERSATION_PROMPT | chat_model() | StrOutputParser()).invoke(question_and_history(state))
    return {"messages": [AIMessage(reply)], "documents": []}


def refuse(state: ChatState) -> dict:
    return {"messages": [AIMessage(REFUSAL)], "documents": []}


@contextmanager
def chat_graph() -> Iterator[CompiledStateGraph]:
    """The compiled graph; it saves to Postgres while the `with` block is open."""
    builder = StateGraph(ChatState)
    builder.add_node(news)
    builder.add_node(conversation)
    builder.add_node(refuse)
    builder.add_conditional_edges(START, route, {"news": "news", "conversation": "conversation", "other": "refuse"})
    with PostgresSaver.from_conn_string(settings.checkpoint_url) as checkpointer:
        checkpointer.setup()  # creates LangGraph's tables if they don't exist
        yield builder.compile(checkpointer=checkpointer)
