"""Кодирование текста в векторы.

Именно этот модуль превращает запросы и документы в embedding'и,
по которым затем работает векторный поиск в Qdrant.
"""

import os
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as functional

# Кеш HuggingFace храним внутри проекта, чтобы не зависеть от глобальной среды пользователя.
DEFAULT_HF_HOME = Path(__file__).resolve().parents[2] / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))

from transformers import AutoModel, AutoTokenizer


class TextEncoder:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        pooling: str = "cls",
        query_prefix: str = "search_query: ",
        document_prefix: str = "search_document: ",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.pooling = pooling
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        # Если CUDA доступна, кодирование пойдет на GPU; иначе на CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pooling(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Усредняем только по настоящим токенам, игнорируя padding.
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        summed = masked_embeddings.sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @staticmethod
    def _cls_pooling(last_hidden_state: torch.Tensor) -> torch.Tensor:
        # Берем embedding первого токена (`[CLS]`) как представление всей последовательности.
        return last_hidden_state[:, 0]

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Разные embedding-модели ожидают разные стратегии pooling.
        if self.pooling == "mean":
            return self._mean_pooling(last_hidden_state, attention_mask)
        return self._cls_pooling(last_hidden_state)

    @staticmethod
    def _normalize_inputs(texts: Iterable[str], prefix: str) -> list[str]:
        # Некоторые retrieval-модели просят разные префиксы для query и document,
        # чтобы точнее разводить два режима использования.
        items = [text.strip() for text in texts if text and text.strip()]
        if not prefix:
            return items
        return [f"{prefix}{text}" for text in items]

    def encode(self, texts: Iterable[str], prefix: str = "") -> list[list[float]]:
        items = self._normalize_inputs(texts, prefix=prefix)
        if not items:
            return []

        vectors: list[list[float]] = []
        with torch.no_grad():
            for index in range(0, len(items), self.batch_size):
                # Батчи снижают расход памяти и позволяют кодировать много чанков за один прогон модели.
                batch = items[index : index + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                pooled = self._pool(outputs.last_hidden_state, encoded["attention_mask"])
                # L2-нормализация особенно полезна для cosine similarity в векторных базах.
                normalized = functional.normalize(pooled, p=2, dim=1)
                vectors.extend(normalized.cpu().tolist())
        return vectors

    def encode_query(self, text: str) -> list[float]:
        # Запрос кодируем отдельно, чтобы применить query-prefix.
        return self.encode([text], prefix=self.query_prefix)[0]

    def encode_documents(self, texts: Iterable[str]) -> list[list[float]]:
        # Документы кодируем отдельно, чтобы применить document-prefix.
        return self.encode(texts, prefix=self.document_prefix)
