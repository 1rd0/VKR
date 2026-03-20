"""Основной orchestration-слой RAG-системы.

Этот класс связывает все этапы пайплайна:
парсинг файлов -> чанкинг -> эмбеддинги -> Qdrant -> LLM/fallback.
"""

from functools import cached_property
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from app.config import Settings
from app.models import DocumentChunk, SearchHit
from app.schemas import AnswerResponse, ChunkPayload, IngestRequest, IngestResponse, SearchResponse
from app.services.chunking import split_text
from app.services.embeddings import TextEncoder
from app.services.llm import AnswerGenerator
from app.services.parsers import extract_text, is_supported_file
from app.services.qdrant_store import QdrantStore


class BaselineRAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Нужные директории подготавливаем сразу при старте,
        # чтобы позже запись файлов не падала из-за отсутствия папок.
        self.settings.raw_dir.mkdir(parents=True, exist_ok=True)
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def encoder(self) -> TextEncoder:
        # Ленивое создание: тяжелая embedding-модель загрузится только когда реально понадобится.
        return TextEncoder(
            model_name=self.settings.embedding_model_name,
            batch_size=self.settings.batch_size,
            pooling=self.settings.embedding_pooling,
            query_prefix=self.settings.embedding_query_prefix,
            document_prefix=self.settings.embedding_document_prefix,
        )

    @cached_property
    def store(self) -> QdrantStore:
        # Аналогично, подключение к Qdrant создаем при первом обращении.
        return QdrantStore(
            url=self.settings.qdrant_url,
            collection_name=self.settings.qdrant_collection,
        )

    @cached_property
    def answer_generator(self) -> AnswerGenerator:
        # Генератор ответа можно переиспользовать между запросами.
        return AnswerGenerator(
            api_key=self.settings.groq_api_key,
            model_name=self.settings.groq_model,
            allow_fallback_answer=self.settings.allow_fallback_answer,
        )

    def health(self) -> dict[str, object]:
        # Health не только говорит "API живо", но и показывает состояние индекса.
        try:
            indexed_chunks = self.store.count()
            status = "ok"
        except Exception:
            indexed_chunks = 0
            status = "degraded"

        return {
            "status": status,
            "collection_name": self.settings.qdrant_collection,
            "indexed_chunks": indexed_chunks,
            "embedding_model": self.settings.embedding_model_name,
            "groq_enabled": self.answer_generator.enabled,
        }

    def ingest(self, request: IngestRequest) -> IngestResponse:
        # Собираем общий список файлов из директории и явных путей.
        files: list[Path] = []
        failed_files: list[str] = []

        if request.directory:
            directory = Path(request.directory)
            if not directory.exists():
                raise FileNotFoundError(f"Directory does not exist: {directory}")
            files.extend(self._collect_files(directory))

        for value in request.paths:
            path = Path(value)
            if path.exists() and is_supported_file(path):
                files.append(path)
            else:
                failed_files.append(value)

        # Убираем дубликаты, сохраняя порядок появления.
        deduplicated_files = list(dict.fromkeys(files))
        chunks_indexed = 0
        files_indexed = 0

        for path in deduplicated_files:
            try:
                # На каждом файле строим список DocumentChunk.
                chunks = self._build_chunks(path)
            except Exception:
                failed_files.append(str(path))
                continue

            if not chunks:
                failed_files.append(str(path))
                continue

            # Сначала превращаем чанки в векторы, затем отправляем пары (chunk, vector) в Qdrant.
            vectors = self.encoder.encode_documents(chunk.text for chunk in chunks)
            self.store.upsert(chunks, vectors)
            chunks_indexed += len(chunks)
            files_indexed += 1

        return IngestResponse(
            files_seen=len(deduplicated_files),
            files_indexed=files_indexed,
            chunks_indexed=chunks_indexed,
            failed_files=sorted(set(failed_files)),
        )

    def search(self, query: str, limit: int | None = None) -> SearchResponse:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if self.store.count() == 0:
            raise IndexError("No index available. Ingest documents first.")
        actual_limit = limit or self.settings.top_k
        # Вопрос пользователя кодируется в тот же векторный space, что и документы.
        query_vector = self.encoder.encode_query(query)
        hits = self.store.search(query_vector=query_vector, limit=actual_limit)
        return SearchResponse(query=query, hits=self._serialize_hits(hits))

    def answer(self, question: str, top_k: int | None = None) -> AnswerResponse:
        if not question.strip():
            raise ValueError("Question must not be empty.")
        actual_top_k = top_k or self.settings.top_k
        # Генерация ответа всегда опирается на retrieval; "ask" не обходит поиск.
        search_response = self.search(query=question, limit=actual_top_k)
        hits = [self._deserialize_hit(hit) for hit in search_response.hits]
        answer, used_llm = self.answer_generator.answer(question=question, hits=hits)
        return AnswerResponse(
            question=question,
            answer=answer,
            used_llm=used_llm,
            hits=search_response.hits,
            meta={
                "top_k": actual_top_k,
                "groq_model": self.settings.groq_model if used_llm else None,
            },
        )

    def save_upload(self, filename: str, content: bytes) -> Path:
        # Берем только basename, чтобы имя файла не могло записать данные вне upload_dir.
        safe_name = Path(filename).name
        destination = self.settings.upload_dir / safe_name
        destination.write_bytes(content)
        return destination

    def _collect_files(self, directory: Path) -> list[Path]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        return sorted(path for path in directory.rglob("*") if is_supported_file(path))

    def _build_chunks(self, path: Path) -> list[DocumentChunk]:
        # 1. Достаем "сырой" текст из файла.
        text = extract_text(path)
        # 2. Режем длинный текст на более короткие перекрывающиеся части.
        chunk_texts = split_text(
            text=text,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        chunks: list[DocumentChunk] = []
        for index, chunk_text in enumerate(chunk_texts):
            # ID зависит от пути, позиции и содержимого чанка:
            # это дает стабильный идентификатор при повторной индексации того же текста.
            identifier = self._make_chunk_id(path=path, index=index, text=chunk_text)
            chunks.append(
                DocumentChunk(
                    point_id=identifier,
                    text=chunk_text,
                    source_path=str(path),
                    metadata={
                        "chunk_index": index,
                        "file_name": path.name,
                        "file_suffix": path.suffix.lower(),
                    },
                )
            )
        return chunks

    @staticmethod
    def _make_chunk_id(path: Path, index: int, text: str) -> str:
        # UUID5 детерминированный: одинаковый вход -> одинаковый ID.
        return str(uuid5(NAMESPACE_URL, f"{path}:{index}:{text}"))

    @staticmethod
    def _serialize_hits(hits: list[SearchHit]) -> list[ChunkPayload]:
        # Преобразуем внутренние SearchHit в Pydantic-схему для API-ответа.
        return [
            ChunkPayload(
                point_id=hit.point_id,
                score=hit.score,
                text=hit.text,
                source_path=hit.source_path,
                metadata=hit.metadata,
            )
            for hit in hits
        ]

    @staticmethod
    def _deserialize_hit(payload: ChunkPayload) -> SearchHit:
        # Обратное преобразование: удобно передавать результаты поиска в AnswerGenerator.
        return SearchHit(
            point_id=payload.point_id,
            score=payload.score,
            text=payload.text,
            source_path=payload.source_path,
            metadata=payload.metadata,
        )
