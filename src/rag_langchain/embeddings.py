"""The embedding model, with nomic-embed-text's task prefixes. (walkthrough steps 8, 9)"""

from functools import cache

from langchain_ollama import OllamaEmbeddings

from rag_langchain.config import settings


class NomicEmbeddings(OllamaEmbeddings):
    """OllamaEmbeddings plus the task prefixes nomic-embed-text was trained with.

    Both sync and async methods are overridden: PGVectorStore's sync methods call the async ones.
    Query methods call super().embed_documents, because OllamaEmbeddings.embed_query calls
    self.embed_documents, which would add the document prefix too.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(["search_document: " + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_documents(["search_query: " + text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await super().aembed_documents(["search_document: " + text for text in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return (await super().aembed_documents(["search_query: " + text]))[0]


@cache
def embeddings() -> NomicEmbeddings:
    return NomicEmbeddings(model=settings.embedding_model)
