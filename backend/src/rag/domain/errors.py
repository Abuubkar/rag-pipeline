class RagError(Exception):
    """Base class for errors the CLI/API should show as a clean message."""


class NoActiveIndex(RagError):
    pass


class EmbedderMismatch(RagError):
    """The query embedder differs from the one the active index was built with."""
