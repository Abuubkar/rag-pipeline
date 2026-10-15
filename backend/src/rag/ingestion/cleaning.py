"""Step 2 of ingestion: normalise text so identical content produces identical chunks.

Bump CLEANING_VERSION whenever the rules change: it is part of the index manifest, so a new
version id (and a fresh index) follows automatically.
"""

import re
import unicodedata

from rag.domain.models import Document
from rag.observability.tracing import annotate, traced

CLEANING_VERSION = 1

_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # e.g. full-width chars, ligatures
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


@traced("ingestion.clean")
def clean_documents(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    cleaned: list[Document] = []
    for doc in docs:
        body = clean_text(doc.body)
        if not body or doc.content_hash in seen:  # drop empty and exact-duplicate articles
            continue
        seen.add(doc.content_hash)
        cleaned.append(doc.model_copy(update={"body": body}))
    annotate(output={"kept": len(cleaned), "dropped": len(docs) - len(cleaned)})
    return cleaned
