"""MultiHop-RAG corpus via Hugging Face `datasets` (yixuantt/MultiHopRAG, ODC-BY).

609 news articles (Sep–Dec 2023). Questions are loaded in Milestone 2.
"""

import hashlib
from datetime import datetime

from datasets import load_dataset

from rag.domain.models import Document


def _doc_id(title: str) -> str:
    # Titles are unique in the corpus and the questions' evidence refers to articles by title,
    # so the id is derived from the title. That lets evaluation map evidence → document ids.
    return "mhr-" + hashlib.sha256(title.encode()).hexdigest()[:16]


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_corpus(name: str, revision: str) -> list[Document]:
    rows = load_dataset(name, "corpus", split="train", revision=revision)
    docs = []
    for row in rows:
        body = row["body"] or ""
        docs.append(
            Document(
                id=_doc_id(row["title"]),
                title=row["title"],
                body=body,
                source=row["source"],
                category=row["category"],
                author=None if row["author"] in (None, "", "None") else row["author"],
                published_at=_parse_date(row["published_at"]),
                url=row["url"],
                content_hash=hashlib.sha256(body.encode()).hexdigest(),
            )
        )
    return docs


def document_id_for_title(title: str) -> str:
    return _doc_id(title)
