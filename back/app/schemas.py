"""Схемы API.

Это контракты между frontend/клиентом и backend: какие поля ожидаем
на входе и что именно возвращаем в ответах.
"""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Ответ health-check: показывает живо ли API и индекс."""

    status: str
    collection_name: str
    indexed_chunks: int
    embedding_model: str
    groq_enabled: bool


class IngestRequest(BaseModel):
    """Что именно индексировать: директорию целиком или конкретные пути."""

    directory: str | None = None
    paths: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    """Итог индексации: сколько файлов и чанков реально попало в индекс."""

    files_seen: int
    files_indexed: int
    chunks_indexed: int
    failed_files: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """Запрос на retrieval без генерации текста."""

    query: str
    limit: int | None = None


class ChunkPayload(BaseModel):
    """Фрагмент документа в API-ответе."""

    point_id: str
    score: float
    text: str
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Ответ поиска с найденными фрагментами."""

    query: str
    hits: list[ChunkPayload]


class AskRequest(BaseModel):
    """Запрос на полноценный ответ с помощью RAG."""

    question: str
    top_k: int | None = None


class AnswerResponse(BaseModel):
    """Ответ RAG: текст + найденные источники + служебные метаданные."""

    question: str
    answer: str
    used_llm: bool
    hits: list[ChunkPayload]
    meta: dict[str, Any] = Field(default_factory=dict)
