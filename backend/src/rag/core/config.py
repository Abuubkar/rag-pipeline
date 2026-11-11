"""Typed configuration: config/settings.yaml, overridden by environment variables (RAG_ prefix)."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

BACKEND_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = BACKEND_DIR / "config"
PROMPTS_DIR = CONFIG_DIR / "prompts"
RESULTS_DIR = BACKEND_DIR.parent / "docs" / "results"


class DatasetConfig(BaseModel):
    name: str
    revision: str


class DatabaseConfig(BaseModel):
    url: str


class OllamaConfig(BaseModel):
    base_url: str


class ChatModelConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.0
    seed: int | None = None
    num_ctx: int | None = None


class LLMConfig(BaseModel):
    generator: ChatModelConfig


class EmbeddingsConfig(BaseModel):
    provider: str
    model: str
    batch_size: int = 32


class ChunkingConfig(BaseModel):
    library: str
    splitter: str
    chunk_size: int
    chunk_overlap: int
    contextual_header: bool = True
    breakpoint_percentile: int = 95  # semantic splitter: split where similarity drops below this percentile


class RetrievalConfig(BaseModel):
    mode: str = "dense"     # dense | keyword | hybrid
    top_k: int = 6          # chunks given to the LLM
    candidates: int = 30    # hybrid: chunks each search brings before fusion
    rrf_k: int = 60         # hybrid: RRF smoothing constant


class IndexConfig(BaseModel):
    alias: str = "default"


class PromptsConfig(BaseModel):
    answer: str = "answer"
    closed_book: str = "closed_book"


class EvaluationConfig(BaseModel):
    dev_fraction: float = 0.2  # share of each question type that goes to dev; the rest is test


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=CONFIG_DIR / "settings.yaml",
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=BACKEND_DIR / ".env",
        extra="ignore",
    )

    dataset: DatasetConfig
    database: DatabaseConfig
    ollama: OllamaConfig
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    index: IndexConfig
    prompts: PromptsConfig
    evaluation: EvaluationConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: explicit args > env vars > .env > settings.yaml
        return (init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls))


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
