"""Dense embeddings with Ollama through LangChain's OllamaEmbeddings.

nomic-embed-text was trained with task prefixes. Documents and queries MUST use different
prefixes, and Ollama does not add them for you. Vectors are L2-normalised so cosine
similarity and dot product agree.
"""

import math

import ollama
from langchain_ollama import OllamaEmbeddings

from rag.domain.models import EmbedderInfo

# Prefixes per model family. Models not listed get no prefix.
_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _model_digest(client: ollama.Client, model: str) -> str:
    wanted = model if ":" in model else f"{model}:latest"
    for m in client.list().models:
        if m.model == wanted:
            return m.digest or "unknown"
    raise RuntimeError(f"Ollama model '{model}' is not pulled. Run: ollama pull {model}")


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self._lc = OllamaEmbeddings(model=model, base_url=base_url)
        self._digest = _model_digest(ollama.Client(host=base_url), model)
        self._doc_prefix, self._query_prefix = _PREFIXES.get(model.split(":")[0], ("", ""))
        self._dims: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._lc.embed_documents([self._doc_prefix + t for t in texts])
        return [_normalize(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return _normalize(self._lc.embed_query(self._query_prefix + text))

    def info(self) -> EmbedderInfo:
        if self._dims is None:
            self._dims = len(self.embed_query("dimension probe"))
        return EmbedderInfo(
            provider="ollama",
            model=self.model,
            model_digest=self._digest,
            dims=self._dims,
            document_prefix=self._doc_prefix,
            query_prefix=self._query_prefix,
            normalize=True,
        )
