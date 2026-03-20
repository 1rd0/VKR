"""Внутренние модели домена.

Эти dataclass'ы не ходят наружу в API, а используются внутри пайплайна
между сервисами: парсингом, поиском, Qdrant и генерацией ответа.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    """Один кусок документа, готовый к векторному поиску."""

    point_id: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    """Результат поиска: найденный chunk и его score."""

    point_id: str
    score: float
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
