"""Composition root: the only place that turns config into concrete objects.

Capabilities never read config to decide *which* vendor to use; they receive ready objects
from here. Swapping Ollama for Groq later is a change in this file and settings.yaml only.
"""

from functools import cached_property
from typing import Any

from rag.core.config import Settings, get_settings
from rag.db import index_registry
from rag.db.session import session_scope
from rag.domain.errors import EmbedderMismatch, NoActiveIndex
from rag.domain.models import IndexVersion
from rag.packages.embeddings.ollama import OllamaEmbedder
from rag.packages.llm.ollama import OllamaChat


class Container:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @property
    def db_url(self) -> str:
        return self.settings.database.url

    # ── Embeddings ──────────────────────────────────────────────────
    @cached_property
    def embedder(self) -> OllamaEmbedder:
        """The embedder named in config: used to BUILD new index versions."""
        cfg = self.settings.embeddings
        if cfg.provider != "ollama":
            raise ValueError(f"Unsupported embeddings provider: {cfg.provider}")
        return OllamaEmbedder(cfg.model, self.settings.ollama.base_url)

    def query_embedder(self, version: IndexVersion) -> OllamaEmbedder:
        """The embedder used to QUERY an index must match the one that built it."""
        built_with = version.manifest.dense
        configured = self.settings.embeddings
        if (configured.provider, configured.model) != (built_with.provider, built_with.model):
            raise EmbedderMismatch(
                f"Index '{version.id}' was built with {built_with.provider}/{built_with.model}, "
                f"but config asks for {configured.provider}/{configured.model}. "
                "Re-run `ingest` with the new model, or switch the active index."
            )
        current = self.embedder.info()
        if current.model_digest != built_with.model_digest:
            raise EmbedderMismatch(
                f"Ollama model '{built_with.model}' changed since index '{version.id}' was built "
                f"(digest {built_with.model_digest[:12]} → {current.model_digest[:12]}). Re-run `ingest`."
            )
        return self.embedder

    # ── Index ───────────────────────────────────────────────────────
    def active_index(self) -> IndexVersion:
        with session_scope(self.db_url) as s:
            version = index_registry.get_active(s, self.settings.index.alias)
        if version is None or version.status != "ready":
            raise NoActiveIndex(
                f"No ready index for alias '{self.settings.index.alias}'. Run `main.py ingest` first."
            )
        return version

    # ── LLM ─────────────────────────────────────────────────────────
    def generator(self, callbacks: list[Any]) -> OllamaChat:
        cfg = self.settings.llm.generator
        if cfg.provider != "ollama":
            raise ValueError(f"Unsupported LLM provider: {cfg.provider}")
        return OllamaChat(
            model=cfg.model,
            base_url=self.settings.ollama.base_url,
            temperature=cfg.temperature,
            seed=cfg.seed,
            num_ctx=cfg.num_ctx,
            callbacks=callbacks,
        )
