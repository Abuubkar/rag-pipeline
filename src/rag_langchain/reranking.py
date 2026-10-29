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
        [
            {"question": question, "query": d.metadata["query"], "excerpt": f"{d.metadata['title']}\n{d.page_content}"}
            for d in documents
        ]
    )
    scored = sorted(zip(documents, [grade.score for grade in grades]), key=lambda pair: pair[1], reverse=True)
    ranked = [document for document, score in scored if score > 0]  # drop unrelated (0)

    # Keep the best chunk of every search, so one part of the question can't take every slot.
    best_of_search = {}
    for document in ranked:
        best_of_search.setdefault(document.metadata["query"], document)
    kept = list(best_of_search.values())
    return kept + [document for document in ranked if document not in kept][: max(0, settings.top_n - len(kept))]
