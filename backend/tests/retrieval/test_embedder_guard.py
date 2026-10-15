"""Querying an index with a different embedding model must be refused.

Vectors from two models live in unrelated spaces; mixing them returns confident garbage
instead of an error, so the container checks the index manifest before every query.
"""

import pytest

from rag.container import Container
from rag.domain.errors import EmbedderMismatch


def test_mismatched_embedding_model_is_refused(container, active_version):
    settings = container.settings.model_copy(deep=True)
    settings.embeddings.model = "mxbai-embed-large"  # not what the index was built with
    with pytest.raises(EmbedderMismatch, match="was built with"):
        Container(settings).query_embedder(active_version)


def test_matching_embedder_is_accepted(container, active_version):
    assert container.query_embedder(active_version).model == active_version.manifest.dense.model
