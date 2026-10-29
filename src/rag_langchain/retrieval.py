"""Find candidate chunks: plan searches → filtered vector search → dedupe. (walkthrough steps 16, 18)"""

from functools import cache

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable, RunnableLambda

from rag_langchain.config import settings
from rag_langchain.llm import chat_model
from rag_langchain.prompts import PLAN_PROMPT
from rag_langchain.vectorstore import publication_names, vector_store


@cache
def planner() -> Runnable:
    names = publication_names()
    # The schema is built from the data: `source` may only be a stored publication name, or null.
    # With method="json_schema" Ollama is forced to follow it, so it can't misspell a name.
    schema = {
        "title": "SearchPlan",
        "type": "object",
        "properties": {
            "searches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "source": {"enum": [*names, None]}},
                    "required": ["query", "source"],
                },
            }
        },
        "required": ["searches"],
    }
    llm = chat_model().with_structured_output(schema, method="json_schema")
    # If the plan can't be parsed, plan no extra searches: the original question is still searched.
    return (PLAN_PROMPT.partial(sources="; ".join(names)) | llm).with_fallbacks(
        [RunnableLambda(lambda _: {"searches": []})],
        exceptions_to_handle=(OutputParserException,),
    )


def search(plan: dict) -> list[Document]:
    metadata_filter = {"source": plan["source"]} if plan.get("source") else None
    return vector_store().similarity_search(plan["query"], k=settings.k_per_search, filter=metadata_filter)


def retrieve(question: str) -> list[Document]:
    # The original question is always searched too, so a poor plan can only add chunks.
    plans = [{"query": question, "source": None}] + planner().invoke({"question": question})["searches"]
    results = RunnableLambda(search).batch(plans)  # all searches in parallel
    unique = {}
    for plan, documents in zip(plans, results):
        for document in documents:
            document.metadata["query"] = plan["query"]  # reranking grades the chunk against its search
            unique.setdefault(document.id, document)  # dedupe by chunk id
    return list(unique.values())
