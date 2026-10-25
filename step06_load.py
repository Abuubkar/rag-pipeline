"""Step 6: load documents.

Every later step (splitting, embedding, storing, retrieving) works on LangChain's `Document`:

    Document(page_content="the text", metadata={"title": ..., "source": ...})

- page_content: the text that will be searched and shown to the LLM
- metadata:     facts *about* the text (for filtering in step 18 and for citations)

A *document loader* turns some source (files, web pages, a dataset) into Documents. Every
LangChain loader implements the same interface, `BaseLoader` from langchain-core:

    lazy_load()  → yields Documents one at a time   (the only method you write)
    load()       → returns them all as a list        (provided by BaseLoader)

LangChain has no maintained loader for Hugging Face datasets (the old one lived in the sunset
`langchain-community` package), so we write our own. It's a few lines, and it shows how every
loader works inside.

Run:  uv run step06_load.py   (first run downloads ~7 MB into ~/.cache/huggingface)
"""

import json
from collections.abc import Iterator

from huggingface_hub import hf_hub_download
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class MultiHopRAGLoader(BaseLoader):
    """Loads the MultiHop-RAG news corpus: one Document per article."""

    def __init__(self, revision: str):
        # A commit hash pins the exact version of the dataset, so results stay reproducible.
        self.revision = revision

    def lazy_load(self) -> Iterator[Document]:
        # Downloads once, then reuses the cached file.
        path = hf_hub_download(
            repo_id="yixuantt/MultiHopRAG",
            repo_type="dataset",
            filename="corpus.json",
            revision=self.revision,
        )
        with open(path, encoding="utf-8") as f:
            articles = json.load(f)  # a list of dicts: title, body, source, category, …

        for article in articles:
            metadata = {key: value for key, value in article.items() if key != "body"}
            # `yield` hands back one Document at a time. That's what makes it "lazy".
            yield Document(page_content=article["body"], metadata=metadata)


loader = MultiHopRAGLoader(revision="71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82")
docs = loader.load()  # we didn't write load(): BaseLoader builds it from lazy_load()

print(f"Loaded {len(docs)} documents")

first = docs[0]
print("\nFirst document")
print("  metadata:    ", first.metadata)
print("  page_content:", first.page_content[:300].replace("\n", " "), "…")

lengths = [len(d.page_content.split()) for d in docs]
print("\nArticle length in words")
print(f"  shortest {min(lengths)} · average {sum(lengths) // len(lengths)} · longest {max(lengths)}")

sources = sorted({d.metadata["source"] for d in docs})
categories = sorted({d.metadata["category"] for d in docs})
print(f"\n{len(sources)} sources, e.g. {sources[:5]}")
print(f"Categories: {categories}")
