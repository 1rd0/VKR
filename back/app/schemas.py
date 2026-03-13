from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    collection_name: str
    indexed_chunks: int
    embedding_model: str
    groq_enabled: bool


class IngestRequest(BaseModel):
    directory: str | None = None
    paths: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    files_seen: int
    files_indexed: int
    chunks_indexed: int
    failed_files: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    limit: int | None = None


class ChunkPayload(BaseModel):
    point_id: str
    score: float
    text: str
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: list[ChunkPayload]


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    used_llm: bool
    hits: list[ChunkPayload]
    meta: dict[str, Any] = Field(default_factory=dict)

