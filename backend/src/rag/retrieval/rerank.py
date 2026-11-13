"""After searching: let the LLM reorder the candidates (retrieval.rerank).

Search ranks each chunk on its own (a vector distance, a keyword score). A listwise reranker
shows the LLM the question and ALL the candidates at once, numbered, and asks for an ordering
("4 > 1 > 12 > ..."). Seeing them side by side lets it judge relevance by reading, and prefer a
set that covers every article a multi-hop question needs. This is the RankGPT approach.

Cost: one LLM call per question that READS every candidate, so passages are trimmed to
`passage_chars` to keep 20 of them inside the context window. Search brings
`rerank_candidates` chunks; after reranking the first `top_k` go to the answer prompt.

The LLM's reply is parsed defensively: unknown or repeated numbers are ignored, and any
candidate it left out keeps its search order after the ones it ranked.
"""

import re
from typing import Callable

from rag.core.prompts import Prompt
from rag.domain.models import ScoredChunk
from rag.domain.state import AnswerState
from rag.observability.tracing import annotate
from rag.packages.llm.ollama import OllamaChat

_NUMBER = re.compile(r"\d+")


def rerank(question: str, chunks: list[ScoredChunk], *, chat: OllamaChat, prompt: Prompt,
           passage_chars: int) -> list[ScoredChunk]:
    """Returns every chunk, reordered; score = 1 / new rank. The caller keeps the top k."""
    if len(chunks) < 2:
        return chunks
    passages = "\n\n".join(
        f"[{n}] {c.title} — {c.source or 'unknown source'}, "
        f"{c.published_at.date().isoformat() if c.published_at else 'unknown date'}\n"
        f"{_trim(c.text, passage_chars)}"
        for n, c in enumerate(chunks, start=1)
    )
    reply = chat.complete(system=prompt.system, user=prompt.render_user(question=question, passages=passages),
                          run_name=f"llm.{prompt.label}").text
    order: list[int] = []
    for number in map(int, _NUMBER.findall(reply)):
        if 1 <= number <= len(chunks) and number - 1 not in order:
            order.append(number - 1)
    order += [i for i in range(len(chunks)) if i not in order]  # left out → keep search order
    return [chunks[i].model_copy(update={"score": 1 / rank}) for rank, i in enumerate(order, start=1)]


def _trim(text: str, max_chars: int) -> str:
    """Cut at a word boundary so the LLM doesn't see half a word."""
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"


def rerank_node(*, top_k: int, **rerank_kwargs) -> Callable[[AnswerState], dict]:
    """rerank_kwargs: everything `rerank` takes except the question and chunks."""
    def node(state: AnswerState) -> dict:
        chunks = rerank(state["question"], state["chunks"], **rerank_kwargs)[:top_k]
        annotate(output=[{"title": c.title} for c in chunks])
        return {"chunks": chunks}

    return node
