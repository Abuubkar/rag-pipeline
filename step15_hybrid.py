"""Step 15: hybrid search (vector + keyword in one query).

Step 14 showed each search failing where the other works: keyword search finds exact names, vector
search understands questions. Hybrid search runs both and merges (fuses) the two result lists:

    question ─┬─► vector search   (top primary_top_k)   ─┐
              └─► keyword search  (top secondary_top_k) ─┴─► fusion ─► top k

Fusion: reciprocal rank fusion (RRF). Each chunk scores 1 / (60 + rank) in every list it appears in,
and the scores are added. It uses only the *ranks*, because the raw scores can't be compared
(cosine distance vs. keyword rank are on different scales). A chunk found by both searches gets two
scores, so it rises to the top.

Keyword-only and vector-only are the same search with one half turned off (top_k=0).

Run:  uv run step15_hybrid.py   (Postgres from step 10 must be running)
"""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion

from step10_postgres import vector_store


def search(question: str, vector_k: int, keyword_k: int) -> list[Document]:
    # A new config on every call (see step 14: a shared config reuses the first query).
    config = HybridSearchConfig(
        primary_top_k=vector_k,
        secondary_top_k=keyword_k,
        fusion_function=reciprocal_rank_fusion,
    )
    return vector_store.similarity_search(question, k=5, hybrid_search_config=config)


hybrid = RunnableLambda(lambda question: search(question, vector_k=10, keyword_k=10))

questions = [
    "Kindle Echo discounts",
    "Sam Bankman-Fried trial",
    "Why is the FTC suing Amazon?",
    "Who is the individual associated with the cryptocurrency industry facing a criminal trial "
    "on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?",
]

for question in questions:
    print(f"\nQ: {question}")
    results = {
        "keyword": search(question, vector_k=0, keyword_k=5),
        "vector": search(question, vector_k=5, keyword_k=0),
        "hybrid": hybrid.invoke(question),
    }
    for name, documents in results.items():
        print(f"  {name}: {len(documents)} results")
        for document in documents:
            print(f"    {document.metadata['title'][:70]}")
