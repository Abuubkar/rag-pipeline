"""Step 14: keyword search (Postgres full-text search).

Vector search matches *meaning*. Keyword search matches *exact words*: names, product names and
rare terms ("Kindle", "FTX") are where it wins, and long natural questions are where it loses.

The classic LangChain choice is BM25Retriever, but it lives in the sunset langchain-community.
PGVectorStore has keyword search built in, using Postgres full-text search:

    to_tsvector('english', text)   'The FTC sued Amazon'           →  'amazon':4 'ftc':2 'su':3
    plainto_tsquery('english', q)  'Why is the FTC suing Amazon?'  →  'ftc' & 'su' & 'amazon'
                                   (stopwords dropped, words reduced to stems, EVERY word must appear)
    ts_rank_cd(...)                rank matches by how often and how close the words appear

It comes as part of HybridSearchConfig (step 15 combines both searches). primary_top_k=0 turns
the vector half off, leaving keyword search only.

Run:  uv run step14_keyword.py   (Postgres from step 10 must be running)
"""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig

from step10_postgres import vector_store


def keyword_search(question: str) -> list[Document]:
    # A new config on every call: PGVectorStore writes the first query into the config it's given
    # and reuses it for every later search, so a shared config keeps returning the first results.
    config = HybridSearchConfig(primary_top_k=0, secondary_top_k=5)  # 0 vector + 5 keyword results
    return vector_store.similarity_search(question, k=5, hybrid_search_config=config)


keyword = RunnableLambda(keyword_search)  # same interface as a retriever: .invoke(question) → list[Document]
vector = vector_store.as_retriever(search_kwargs={"k": 5})

questions = [
    "Kindle Echo discounts",
    "Sam Bankman-Fried trial",
    "Why is the FTC suing Amazon?",
    "Who is the individual associated with the cryptocurrency industry facing a criminal trial "
    "on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?",
]

for question in questions:
    print(f"\nQ: {question}")
    for name, retriever in [("keyword", keyword), ("vector", vector)]:
        documents = retriever.invoke(question)
        print(f"  {name}: {len(documents)} results")
        for document in documents:
            print(f"    {document.metadata['title'][:70]}")
