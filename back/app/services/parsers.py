"""Извлечение текста из файлов разных форматов."""

from pathlib import Path

import fitz
from bs4 import BeautifulSoup


# Поддерживаем только форматы, для которых умеем извлекать текст без OCR.
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md"}


def is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def read_text_file(path: Path) -> str:
    # Пробуем несколько типичных кодировок, потому что документы могут быть не только UTF-8.
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="ignore")


def extract_text(path: Path) -> str:
    """Выбирает нужный способ парсинга по расширению файла."""

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix in {".html", ".htm"}:
        return extract_html_text(path)
    return read_text_file(path)


def extract_pdf_text(path: Path) -> str:
    # `fitz` читает PDF постранично; затем мы склеиваем все страницы в один текст.
    with fitz.open(path) as document:
        pages = [page.get_text("text") for page in document]
        return "\n".join(pages)


def extract_html_text(path: Path) -> str:
    # У HTML убираем разметку и оставляем только видимый текст.
    html = read_text_file(path)
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n", strip=True)
