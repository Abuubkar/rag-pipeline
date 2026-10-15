"""Graph state: the data that flows between answer-graph nodes.

Each node reads some keys and returns a dict with the keys it updates.
"""

from typing import TypedDict

from rag.domain.models import Citation, Completion, ScoredChunk


class AnswerState(TypedDict, total=False):
    # input
    question: str
    # retrieval.search
    chunks: list[ScoredChunk]
    # generation.context
    context: str
    # generation.generate
    completion: Completion
    prompt: str  # name@version
    # generation.citations
    citations: list[Citation]
    invalid_citations: list[int]
