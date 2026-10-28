"""Loads the MultiHop-RAG news corpus as LangChain Documents. (walkthrough step 6)"""

import json
from collections.abc import Iterator

from huggingface_hub import hf_hub_download
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class MultiHopRAGLoader(BaseLoader):
    """One Document per article: the body as page_content, everything else as metadata."""

    def __init__(self, revision: str):
        self.revision = revision  # a commit hash pins the exact dataset version

    def lazy_load(self) -> Iterator[Document]:
        path = hf_hub_download(
            repo_id="yixuantt/MultiHopRAG",
            repo_type="dataset",
            filename="corpus.json",
            revision=self.revision,
        )
        with open(path, encoding="utf-8") as f:
            articles = json.load(f)

        for article in articles:
            metadata = {key: value for key, value in article.items() if key != "body"}
            yield Document(page_content=article["body"], metadata=metadata)
