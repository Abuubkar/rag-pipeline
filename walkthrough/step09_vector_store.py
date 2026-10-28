"""Step 9: in-memory vector store.

A vector store keeps (text, metadata, vector) together and answers "which stored texts are most
similar to this query?". It calls the embedding model for you: once per chunk when you add
documents, and once per question when you search. No hand-written cosine similarity needed.

InMemoryVectorStore keeps everything in a Python dict, so it's rebuilt on every run.
Step 10 moves to Postgres so the vectors are saved.

Run:  uv run walkthrough/step09_vector_store.py
"""

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from step06_load import REVISION, MultiHopRAGLoader
from step07_split import splitter


class NomicEmbeddings(OllamaEmbeddings):
    """OllamaEmbeddings that adds the task prefixes nomic-embed-text was trained with.

    The vector store calls these methods itself, so the prefixes have to live here:
    - The async versions too: PGVectorStore's sync methods call aembed_documents / aembed_query
      internally, so overriding only the sync ones does nothing there.
    - The query methods call super().embed_documents, because OllamaEmbeddings.embed_query calls
      self.embed_documents, which would add "search_document: " on top of "search_query: ".
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(["search_document: " + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_documents(["search_query: " + text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await super().aembed_documents(["search_document: " + text for text in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return (await super().aembed_documents(["search_query: " + text]))[0]


# Runs only with `uv run walkthrough/step09_vector_store.py`, not when another step imports NomicEmbeddings.
if __name__ == "__main__":
    articles = MultiHopRAGLoader(revision=REVISION).load()[:30]
    chunks = splitter.split_documents(articles)

    vector_store = InMemoryVectorStore(NomicEmbeddings(model="nomic-embed-text"))
    ids = vector_store.add_documents(chunks)  # embeds every chunk and stores it; returns one id per chunk
    print(f"Stored {len(ids)} chunks from {len(articles)} articles, e.g. id {ids[0]}")

    for question in ["Where can I get discounts on Kindle and Echo devices?", "Why is the FTC suing Amazon?"]:
        print(f"\nQ: {question}")
        for chunk, score in vector_store.similarity_search_with_score(question, k=5):
            print(f"  {score:.3f}  {chunk.metadata['title'][:60]}")
