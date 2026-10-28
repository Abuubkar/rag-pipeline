"""Step 17: reranking (let the LLM judge each chunk).

Step 16 returns 8–12 chunks in no useful order: just "original question's chunks first, then each
query's". Some are noise. Searches can only say "these words/vectors are close"; they can't tell
whether a chunk actually helps answer the question. An LLM can.

Reranking: the LLM reads the question + ONE chunk and gives it a score from 0 to 10. All chunks are
scored in parallel with .batch(), sorted by score, and the top 5 are kept:

    question ─► multi-query (step 16) ─► 12 chunks ─► LLM scores each ─► sort ─► top 5

This is "pointwise" reranking: each chunk is judged on its own. (The legacy langchain-classic
LLMListwiseRerank sends all chunks in one prompt and asks for an order instead.)

Run:  uv run walkthrough/step17_rerank.py   (Postgres from step 10 and Ollama must be running)
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from step16_multi_query import chain as multi_query
from step16_multi_query import llm

TOP_N = 5


class Relevance(BaseModel):
    score: int


# method="json_schema" constrains the output to the schema, so it can't fail on a misnamed tool like
# step 16's function_calling did. But the model never sees field descriptions in this mode (step 5),
# so the scoring scale goes in the prompt instead.
grader = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade how useful a news excerpt is for answering a question. Reply with a score from 0 to 10:\n"
            "10 = it directly answers the question or one part of it\n"
            "5 = same topic, but doesn't answer it\n"
            "0 = unrelated",
        ),
        ("human", "Question: {question}\n\nExcerpt:\n{excerpt}"),
    ]
) | llm.with_structured_output(Relevance, method="json_schema")


def grade_documents(inputs: dict) -> list[tuple[Document, int]]:
    documents = inputs["documents"]
    grades = grader.batch(
        [
            {"question": inputs["question"], "excerpt": f"{d.metadata['title']}\n{d.page_content}"}
            for d in documents
        ]
    )
    return sorted(zip(documents, [g.score for g in grades]), key=lambda pair: pair[1], reverse=True)


chain = (
    multi_query                                                                        # question, queries, documents
    .assign(graded=grade_documents)                                                    # + graded: [(doc, score)] best first
    .assign(top=lambda inputs: [document for document, _ in inputs["graded"][:TOP_N]])  # + top: best 5 docs
)

questions = [
    "Who is the individual associated with the cryptocurrency industry facing a criminal trial "
    "on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?",
    "Does 'The Age' article suggest that Australia's Davis Cup team is aiming for an improvement in "
    "their performance compared to the previous year, while the 'Sporting News' article indicates that "
    "the South Africa national rugby team has already achieved an improvement to reach the Rugby World "
    "Cup semi-finals?",
]

for question in questions:
    result = chain.invoke({"question": question})
    print(f"\nQ: {question}")
    print(f"  {len(result['documents'])} chunks from multi-query, scored by the LLM (✅ = kept in the top {TOP_N}):")
    for n, (document, score) in enumerate(result["graded"]):
        mark = "✅" if n < TOP_N else "  "
        print(f"  {mark} {score:2}  {document.metadata['source'][:20]:20}  {document.metadata['title'][:55]}")
