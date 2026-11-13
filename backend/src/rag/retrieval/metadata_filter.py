"""Before searching: turn what the question says about *metadata* into a SQL filter
(retrieval.metadata_filter).

none    search every chunk.
source  search only chunks from the publications the question names ("… reported by TechCrunch
        and The Verge"). Publication names are metadata, not chunk text, so neither dense nor
        keyword search can use them; a WHERE on documents.source can.

Rule-based, not an LLM: the corpus has a short, fixed list of sources, so a case-sensitive,
whole-word match of each name finds them for free. (An LLM "self-query" step that writes the
filter is the general version, for open-ended fields like dates or authors.)
No source named → no filter, so a question never ends up with zero candidates.
"""

import re
from functools import lru_cache

from rag.db.session import session_scope
from rag.packages.vectorstore import pgvector

FILTERS = ("none", "source")


@lru_cache
def known_sources(db_url: str) -> tuple[str, ...]:
    with session_scope(db_url) as s:
        return tuple(pgvector.list_sources(s))


def match_sources(text: str, sources: tuple[str, ...]) -> list[str]:
    # Case-sensitive, so the word "rivals" doesn't select the source "Rivals". The lookarounds
    # keep it to whole names: "The Age" must not match inside "The Agenda".
    return [s for s in sources if re.search(rf"(?<!\w){re.escape(s)}(?!\w)", text)]


def sources_for(text: str, *, mode: str, db_url: str) -> list[str]:
    if mode == "none":
        return []
    if mode not in FILTERS:
        raise ValueError(f"Unknown metadata filter: {mode} ({' | '.join(FILTERS)})")
    return match_sources(text, known_sources(db_url))
