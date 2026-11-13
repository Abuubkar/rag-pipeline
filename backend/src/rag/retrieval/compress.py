"""After searching (and reranking): cut the sentences the question doesn't need
(retrieval.compression).

none       give the LLM the top_k chunks as they are.
embedding  split each chunk into sentences, embed every sentence (as a document) and the question
           (as a query), and keep the `compression_keep` share of sentences most similar to the
           question, across all top_k chunks, in their original order. A chunk that's mostly off
           topic shrinks a lot; a focused one keeps most of its text. Every chunk keeps at least
           its best sentence, so the [n] numbering and citations still work.

Why: fewer tokens is a cheaper, faster prompt, and less distraction for the LLM ("lost in the
middle"). The risk: a sentence that looks unrelated to the question can still hold a fact the
answer needs, and fact recall shows exactly that.

Cost: one batch embedding call per question (≈ 100 sentences), no LLM. The next rung up is an
LLM extractor (LangChain's LLMChainExtractor): smarter, but one LLM call per chunk.
"""

import re
from typing import Callable

from rag.domain.models import ScoredChunk
from rag.domain.state import AnswerState
from rag.observability.tracing import annotate
from rag.packages.chunking.langchain import count_tokens
from rag.packages.embeddings.ollama import OllamaEmbedder

COMPRESSIONS = ("none", "embedding")
# A sentence ends at . ! ? (optionally followed by a closing quote) before whitespace, or at a
# line break. Crude next to a trained splitter (it splits after "Mr."), but cheap and good enough
# to choose which text to drop.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"'”’]?\s+|\n+")


def compress(question: str, chunks: list[ScoredChunk], *, mode: str, embedder: OllamaEmbedder | None,
             keep: float) -> list[ScoredChunk]:
    if mode == "none" or not chunks:
        return chunks
    if mode not in COMPRESSIONS:
        raise ValueError(f"Unknown compression: {mode} ({' | '.join(COMPRESSIONS)})")
    assert embedder
    sentences = [[s.strip() for s in _SENTENCE_END.split(c.text) if s.strip()] for c in chunks]
    flat = [(ci, si, s) for ci, chunk_sentences in enumerate(sentences) for si, s in enumerate(chunk_sentences)]
    q = embedder.embed_query(question)
    vectors = embedder.embed_documents([s for _, _, s in flat])
    similarity = {(ci, si): sum(a * b for a, b in zip(q, v)) for (ci, si, _), v in zip(flat, vectors)}  # normalized → cosine

    n_keep = max(1, round(len(flat) * keep))
    kept = set(sorted(similarity, key=similarity.get, reverse=True)[:n_keep])  # type: ignore[arg-type]
    for ci, chunk_sentences in enumerate(sentences):  # every chunk keeps its best sentence
        if chunk_sentences and not any((ci, si) in kept for si in range(len(chunk_sentences))):
            kept.add(max(((ci, si) for si in range(len(chunk_sentences))), key=similarity.get))  # type: ignore[arg-type]

    compressed = []
    for ci, chunk in enumerate(chunks):
        text = " ".join(s for si, s in enumerate(sentences[ci]) if (ci, si) in kept)
        compressed.append(chunk.model_copy(update={"text": text, "token_count": count_tokens(text)}))
    return compressed


def compress_node(**compress_kwargs) -> Callable[[AnswerState], dict]:
    """compress_kwargs: everything `compress` takes except the question and chunks."""
    def node(state: AnswerState) -> dict:
        before = sum(c.token_count for c in state["chunks"])
        chunks = compress(state["question"], state["chunks"], **compress_kwargs)
        annotate(output={"tokens_before": before, "tokens_after": sum(c.token_count for c in chunks)})
        return {"chunks": chunks}

    return node
