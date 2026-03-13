from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "rag-baseline"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_baseline"
    embedding_model_name: str = "ai-forever/ru-en-RoSBERTa"
    embedding_pooling: str = "cls"
    embedding_query_prefix: str = "search_query: "
    embedding_document_prefix: str = "search_document: "
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150
    batch_size: int = 16
    data_dir: Path = Field(default=PROJECT_ROOT / "back" / "data")
    allow_fallback_answer: bool = True

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
