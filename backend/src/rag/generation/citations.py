"""Turn [n] markers in the answer into Citation objects, and flag numbers that point nowhere.

This only checks that a citation *exists*. Whether the cited source actually *supports* the
sentence is a groundedness check (Milestone 4).
"""

import re

from rag.domain.models import Citation, ScoredChunk
from rag.domain.state import AnswerState
from rag.observability.tracing import annotate

# Matches [3] and grouped forms like [1, 6] or [2,4,5].
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(text: str, sources: list[ScoredChunk]) -> tuple[list[Citation], list[int]]:
    numbers = sorted({int(n) for group in _MARKER.findall(text) for n in group.split(",")})
    citations, invalid = [], []
    for n in numbers:
        if 1 <= n <= len(sources):
            s = sources[n - 1]
            citations.append(Citation(number=n, chunk_id=s.chunk_id, document_id=s.document_id,
                                      title=s.title, source=s.source, url=s.url))
        else:
            invalid.append(n)
    return citations, invalid


def citations_node(state: AnswerState) -> dict:
    citations, invalid = extract_citations(state["completion"].text, state["chunks"])
    annotate(output={"cited": [c.number for c in citations], "invalid": invalid})
    return {"citations": citations, "invalid_citations": invalid}
