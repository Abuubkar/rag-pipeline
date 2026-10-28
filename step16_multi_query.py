"""Step 16: multi-query (ask the question several ways).

One long question becomes one vector, and that vector lands near ONE topic. A multi-hop question
("…as reported by both The Verge and TechCrunch", "Australia's Davis Cup team … while South Africa's
rugby team …") needs chunks about several things, so a single search misses some of them (step 12).

Multi-query: the LLM splits the question into short search queries, each query is searched
separately, and the results are merged (duplicates removed):

    question ─► LLM ─► [question, "query 1", "query 2", …] ─► retriever on each ─► merge ─► documents

Built from parts we already know: structured output (step 5), a retriever (step 11),
.assign (step 13). New: `retriever.map()`, which runs a Runnable on every item of a list.
(LangChain's ready-made MultiQueryRetriever is in the legacy langchain-classic; see ROADMAP.)

Run:  uv run step16_multi_query.py   (Postgres from step 10 and Ollama must be running)
"""

from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from step10_postgres import vector_store

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
llm = ChatOllama(model="gemma4:12b", temperature=0, reasoning=False)


class SearchQueries(BaseModel):
    queries: list[str] = Field(
        description="3 short search queries, each about ONE part of the question (a person, event or team)"
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write queries for a news search engine. Split the user's question into its separate "
            "parts and write one short query for each part.",
        ),
        ("human", "{question}"),
    ]
)
# The model sometimes names its tool call "search_queries" instead of "SearchQueries", and LangChain
# raises OutputParserException instead of accepting it. with_fallbacks catches that error and runs the
# fallback instead: no extra queries, so only the original question is searched.
query_writer = (prompt | llm.with_structured_output(SearchQueries, method="function_calling")).with_fallbacks(
    [RunnableLambda(lambda inputs: SearchQueries(queries=[]))],
    exceptions_to_handle=(OutputParserException,),
)


def unique_documents(document_lists: list[list[Document]]) -> list[Document]:
    # Several queries often find the same chunk: keep the first copy.
    seen, unique = set(), []
    for documents in document_lists:
        for document in documents:
            if document.id not in seen:
                seen.add(document.id)
                unique.append(document)
    return unique


chain = (
    # The original question is searched too, so bad LLM queries can only add chunks, never lose them.
    RunnablePassthrough.assign(queries=lambda inputs: [inputs["question"]] + query_writer.invoke(inputs).queries)
    .assign(documents=itemgetter("queries") | retriever.map() | unique_documents)  # + documents
)

# Runs only with `uv run step16_multi_query.py`, not when another step imports the chain.
if __name__ == "__main__":
    questions = [
        "Who is the individual associated with the cryptocurrency industry facing a criminal trial "
        "on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?",
        # From the MultiHop-RAG question set. Needs one article from The Age, one from Sporting News.
        "Does 'The Age' article suggest that Australia's Davis Cup team is aiming for an improvement in "
        "their performance compared to the previous year, while the 'Sporting News' article indicates that "
        "the South Africa national rugby team has already achieved an improvement to reach the Rugby World "
        "Cup semi-finals?",
        # Makes the query writer fail (wrong tool name), so the fallback runs.
        "Which online betting platform provides a welcome bonus of up to $1000 in bonus bets for new customers' "
        "first losses, runs NBA betting promotions, and is anticipated to extend the same sign-up offer to new "
        "users in Vermont, as reported by both CBSSports.com and Sporting News?",
    ]

    for question in questions:
        print(f"\nQ: {question}")

        single = retriever.invoke(question)
        print(f"\n  one search: {len(single)} chunks")
        for document in single:
            print(f"    {document.metadata['source'][:20]:20}  {document.metadata['title'][:60]}")

        result = chain.invoke({"question": question})
        print(f"\n  multi-query: {result['queries']}")
        print(f"  → {len(result['documents'])} unique chunks")
        for document in result["documents"]:
            print(f"    {document.metadata['source'][:20]:20}  {document.metadata['title'][:60]}")
