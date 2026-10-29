"""Answer a question: rewrite follow-ups → retrieve → rerank → answer with numbered citations and chat history.
(walkthrough steps 12, 13, 19, 20)"""

from functools import cache
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from rag_langchain.llm import chat_model
from rag_langchain.prompts import ANSWER_PROMPT
from rag_langchain.reranking import rerank
from rag_langchain.retrieval import retrieve, standalone_question


def format_docs(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{n}] {d.metadata['title']} — {d.metadata['source']}\n{d.page_content}"
        for n, d in enumerate(documents, start=1)
    )


@cache
def rag_chain() -> Runnable:
    return (
        RunnablePassthrough.assign(search_question=standalone_question)  # + search_question
        .assign(candidates=itemgetter("search_question") | RunnableLambda(retrieve))  # + candidates
        .assign(documents=lambda inputs: rerank(inputs["search_question"], inputs["candidates"]))  # + documents (top N)
        .assign(context=lambda inputs: format_docs(inputs["documents"]))  # + context
        .assign(answer=ANSWER_PROMPT | chat_model() | StrOutputParser())  # + answer
    )


def answer(question: str, history: list[BaseMessage] | None = None) -> dict:
    """Returns {"question", "history", "search_question", "candidates", "documents", "context", "answer"}."""
    return rag_chain().invoke({"question": question, "history": history or []})
