"""Step 18: metadata filtering ("only TechCrunch articles").

Step 16's queries put publication names in the query TEXT ('"Sporting News" South Africa rugby…'). Text
only nudges the vector, so other publications still came back, and for the betting question the
CBSSports.com article never came back at all.

Every chunk already carries its publication in metadata ({"source": "CBSSports.com", …}). A *filter*
makes Postgres search only the chunks whose metadata matches:

    vector_store.similarity_search("betting bonus", filter={"source": "CBSSports.com"})
    → WHERE langchain_metadata->>'source' = 'CBSSports.com'   (then the vector search)

The LLM plans the searches: one per part of the question, each with a publication if the question
names one. `source` is a Literal of the 49 real publication names, so with method="json_schema"
Ollama can only produce an exact, valid name (no typos, no misnamed tool as in step 16).
(LangChain's ready-made SelfQueryRetriever does this too, but lives in the legacy langchain-classic.)

    question ─► LLM plan: [(query, source), …] ─► filtered search for each ─► merge ─► documents

Run:  uv run step18_metadata_filter.py   (Postgres from step 10 and Ollama must be running)
"""

from operator import itemgetter
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel

from step06_load import REVISION, MultiHopRAGLoader
from step10_postgres import vector_store
from step16_multi_query import llm, unique_documents

# The exact publication names stored in metadata (49 of them).
SOURCES = sorted({document.metadata["source"] for document in MultiHopRAGLoader(revision=REVISION).load()})

class Search(BaseModel):
    query: str
    source: Literal[tuple(SOURCES)] | None = None  # only these exact names, or None for "any publication"


class SearchPlan(BaseModel):
    searches: list[Search]


# json_schema hides field descriptions from the model (step 5), so the rules go in the prompt.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You plan searches for a news search engine. Split the user's question into its separate parts "
            "and write one short search query for each part. If the question says which publication reported "
            "a part, set that search's source to the publication's name; otherwise set source to null.\n\n"
            "Publications: {sources}",
        ),
        ("human", "{question}"),
    ]
).partial(sources="; ".join(SOURCES))
planner = prompt | llm.with_structured_output(SearchPlan, method="json_schema")


def search(plan: Search) -> list[Document]:
    metadata_filter = {"source": plan.source} if plan.source else None
    return vector_store.similarity_search(plan.query, k=5, filter=metadata_filter)


chain = (
    # The original question is searched too, unfiltered (as in step 16).
    RunnablePassthrough.assign(searches=lambda inputs: [Search(query=inputs["question"])] + planner.invoke(inputs).searches)
    .assign(documents=itemgetter("searches") | RunnableLambda(search).map() | unique_documents)
)

# Runs only with `uv run step18_metadata_filter.py`, not when another step imports the chain.
if __name__ == "__main__":
    questions = [
        "Does 'The Age' article suggest that Australia's Davis Cup team is aiming for an improvement in "
        "their performance compared to the previous year, while the 'Sporting News' article indicates that "
        "the South Africa national rugby team has already achieved an improvement to reach the Rugby World "
        "Cup semi-finals?",
        "Which online betting platform provides a welcome bonus of up to $1000 in bonus bets for new customers' "
        "first losses, runs NBA betting promotions, and is anticipated to extend the same sign-up offer to new "
        "users in Vermont, as reported by both CBSSports.com and Sporting News?",
    ]

    for question in questions:
        result = chain.invoke({"question": question})
        print(f"\nQ: {question}")
        print("  searches:")
        for plan in result["searches"]:
            print(f"    source={plan.source!r:30} query={plan.query!r}")
        print(f"  → {len(result['documents'])} unique chunks")
        for document in result["documents"]:
            print(f"    {document.metadata['source'][:20]:20}  {document.metadata['title'][:60]}")
