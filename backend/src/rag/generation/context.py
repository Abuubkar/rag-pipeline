"""Format retrieved chunks as numbered sources. The numbers are what the model cites as [n]."""

from rag.domain.models import ScoredChunk
from rag.domain.state import AnswerState


def format_sources(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for n, chunk in enumerate(chunks, start=1):
        date = chunk.published_at.date().isoformat() if chunk.published_at else "unknown date"
        header = f"[{n}] {chunk.title} — {chunk.source or 'unknown source'}, {date}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def context_node(state: AnswerState) -> dict:
    return {"context": format_sources(state["chunks"])}
