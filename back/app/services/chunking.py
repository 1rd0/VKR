"""Подготовка текста к индексации.

Задача этого модуля: превратить длинный документ в последовательность
перекрывающихся кусков, которые удобно кодировать в эмбеддинги.
"""

import re


def normalize_text(text: str) -> str:
    """Приводит текст к более стабильному виду перед чанкингом."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Режет текст на чанки примерно одинакового размера.

    `chunk_overlap` нужен, чтобы информация на границе чанков не терялась:
    хвост предыдущего куска частично повторяется в следующем.
    """

    cleaned = normalize_text(text)
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            # Пытаемся резать не посреди слова, а ближе к "естественной" границе:
            # точке, переводу строки или хотя бы пробелу.
            sentence_break = max(
                cleaned.rfind(". ", start, end),
                cleaned.rfind("! ", start, end),
                cleaned.rfind("? ", start, end),
                cleaned.rfind("\n", start, end),
                cleaned.rfind(" ", start, end),
            )
            if sentence_break > start + chunk_size // 2:
                end = sentence_break + 1

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Двигаем старт назад на overlap-символов, чтобы соседние чанки пересекались.
        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    return chunks
