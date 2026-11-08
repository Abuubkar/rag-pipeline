"""MultiHop-RAG corpus via Hugging Face `datasets` (yixuantt/MultiHopRAG, ODC-BY).

609 news articles (Sep–Dec 2023) and 2,556 questions with gold evidence.
"""

import hashlib
from datetime import datetime

from datasets import load_dataset

from rag.domain.models import Document, GoldenItem


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


def load_questions(name: str, revision: str) -> list[GoldenItem]:
    rows = load_dataset(name, "MultiHopRAG", split="train", revision=revision)
    items = []
    for row in rows:
        evidence = row["evidence_list"]
        items.append(
            GoldenItem(
                id="q-" + hashlib.sha256(row["query"].encode()).hexdigest()[:16],
                question=row["query"],
                answer=row["answer"],
                question_type=row["question_type"],
                # One article can back several facts, so keep each id once, in order.
                evidence_document_ids=list(dict.fromkeys(_doc_id(e["title"]) for e in evidence)),
                evidence_facts=[e["fact"] for e in evidence],
                answerable=row["question_type"] != "null_query",
            )
        )
    return items
