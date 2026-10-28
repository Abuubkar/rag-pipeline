"""Keep the chunks that best answer the question: the LLM scores each one. (walkthrough step 17)"""

from functools import cache

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from rag_langchain.config import settings
from rag_langchain.llm import chat_model
from rag_langchain.prompts import GRADE_PROMPT


class Relevance(BaseModel):
    score: int


@cache
def grader() -> Runnable:
    # json_schema: Ollama forces the {"score": int} shape, so it can't fail on a misnamed tool.
    return GRADE_PROMPT | chat_model().with_structured_output(Relevance, method="json_schema")


def rerank(question: str, documents: list[Document]) -> list[Document]:
    grades = grader().batch(  # one LLM call per chunk, run in parallel
        [{"question": question, "excerpt": f"{d.metadata['title']}\n{d.page_content}"} for d in documents]
    )
    scored = sorted(zip(documents, [grade.score for grade in grades]), key=lambda pair: pair[1], reverse=True)
    return [document for document, score in scored[: settings.top_n] if score > 0]  # drop unrelated (0)
