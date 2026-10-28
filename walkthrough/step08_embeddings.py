"""Step 8: embeddings.

An embedding model turns text into a vector: a list of numbers (768 for nomic-embed-text).
Texts with similar *meaning* get vectors that point in similar directions, even when they share
no words. Cosine similarity measures that: 1.0 = same direction, ~0 = unrelated.

This is how RAG finds relevant chunks: embed every chunk once, embed the question, and pick the
chunks whose vectors are closest to the question's.

Run:  uv run walkthrough/step08_embeddings.py   (Ollama must be running, with `ollama pull nomic-embed-text`)
"""

import math

from langchain_ollama import OllamaEmbeddings

from step06_load import REVISION, MultiHopRAGLoader
from step07_split import splitter

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# nomic-embed-text was trained with task prefixes: one for documents, another for queries.
# LangChain's OllamaEmbeddings does not add them, so we do.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(A, B) = A·B / (|A| x |B|): the dot product divided by the product of the vectors' lengths."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


vector = embeddings.embed_query(QUERY_PREFIX + "FTX trial")
print(f"One text → one vector of {len(vector)} numbers: {[round(v, 3) for v in vector[:5]]} …")

articles = MultiHopRAGLoader(revision=REVISION).load()[:20] # first 30 articles
sample_chunks = splitter.split_documents(articles)  # all chunks of the 30 articles (~400)

# embed_documents takes a list and returns one vector per text, in the same order.
chunk_vectors = embeddings.embed_documents([DOCUMENT_PREFIX + c.page_content for c in sample_chunks])

for question in ["Where can I get discounts on Kindle and Echo devices?", "Why is the FTC suing Amazon?"]:
    question_vector = embeddings.embed_query(QUERY_PREFIX + question)
    scored = sorted(
        ((cosine_similarity(question_vector, v), c.metadata["title"]) for c, v in zip(sample_chunks, chunk_vectors)),
        reverse=True,
    )
    print(f"\nQ: {question}")
    for score, title in scored[:5]:  # top 5 chunks
        print(f"  {score:.3f}  {title[:70]}")
