"""Step 1 of ingestion: get the source documents."""

from rag.domain.models import Document
from rag.observability.tracing import annotate, traced
from rag.packages.documents import multihop_rag


@traced("ingestion.load")
def load_documents(dataset: str, revision: str, limit: int | None = None) -> list[Document]:
    docs = multihop_rag.load_corpus(dataset, revision)
    if limit:
        docs = docs[:limit]
    annotate(output={"documents": len(docs), "dataset": dataset, "revision": revision[:10]})
    return docs
