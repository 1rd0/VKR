"""Адаптер для работы с Qdrant.

Этот слой изолирует детали векторной БД от остального приложения:
сервис RAG просто отдает сюда чанки и векторы.
"""

from typing import Any

from qdrant_client import QdrantClient, models

from app.models import DocumentChunk, SearchHit


class QdrantStore:
    def __init__(self, url: str, collection_name: str) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)

    def collection_exists(self) -> bool:
        # В новых версиях клиента есть удобный `collection_exists`,
        # но оставляем fallback для совместимости.
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(self.collection_name))

        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def ensure_collection(self, vector_size: int) -> None:
        # Коллекция создается лениво, только когда мы впервые знаем размер вектора.
        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if not chunks or not vectors:
            return

        self.ensure_collection(vector_size=len(vectors[0]))
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            # Основной текст и путь кладем в payload, чтобы потом вернуть их в ответе поиска.
            payload = {
                "text": chunk.text,
                "source_path": chunk.source_path,
                **chunk.metadata,
            }
            points.append(
                models.PointStruct(
                    id=chunk.point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], limit: int) -> list[SearchHit]:
        if not self.collection_exists():
            return []

        # Qdrant возвращает ближайшие вектора по cosine distance.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        points = getattr(response, "points", response)

        hits: list[SearchHit] = []
        for point in points:
            # Нормализуем ответ Qdrant к нашей внутренней модели SearchHit,
            # чтобы остальная часть системы не зависела от формата клиента.
            payload: dict[str, Any] = dict(point.payload or {})
            hits.append(
                SearchHit(
                    point_id=str(point.id),
                    score=float(getattr(point, "score", 0.0) or 0.0),
                    text=str(payload.pop("text", "")),
                    source_path=str(payload.pop("source_path", "unknown")),
                    metadata=payload,
                )
            )
        return hits

    def count(self) -> int:
        # Используется в health-check и чтобы понять, готов ли индекс к поиску.
        if not self.collection_exists():
            return 0
        result = self.client.count(collection_name=self.collection_name, exact=True)
        return int(result.count)
