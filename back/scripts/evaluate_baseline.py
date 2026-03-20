"""Локальная оценка retrieval-качества baseline API.

Скрипт ходит в уже запущенный backend, вызывает `/search` по тестовым вопросам
и считает базовые метрики поиска по заранее размеченному CSV.
"""

import argparse
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path


BACK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_PATH = BACK_ROOT / "eval" / "eval_retrieval.csv"
DEFAULT_CATALOG_PATH = BACK_ROOT / "data" / "catalog.csv"
DEFAULT_RESULTS_DIR = BACK_ROOT / "eval" / "results"
DEFAULT_API_URL = "http://127.0.0.1:8000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline retrieval metrics.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL for the local API.")
    parser.add_argument(
        "--eval-path",
        default=str(DEFAULT_EVAL_PATH),
        help="Path to eval_retrieval.csv.",
    )
    parser.add_argument(
        "--catalog-path",
        default=str(DEFAULT_CATALOG_PATH),
        help="Path to catalog.csv.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_RESULTS_DIR / "retrieval_report.csv"),
        help="Where to write the detailed report.",
    )
    return parser.parse_args()


def split_semicolon(value: str) -> list[str]:
    # В CSV несколько значений хранятся в одной ячейке через `;`.
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def load_catalog(catalog_path: Path) -> dict[str, Path]:
    # Каталог связывает логический `doc_id` из eval-файла с реальным путем к документу.
    mapping: dict[str, Path] = {}
    with catalog_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc_id = row["doc_id"].strip()
            file_path = row["file_path"].strip()
            mapping[doc_id] = (BACK_ROOT / ".." / file_path).resolve()
    return mapping


def load_eval_rows(eval_path: Path) -> list[dict[str, str]]:
    with eval_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def get_health(api_url: str) -> dict[str, object]:
    # Перед началом оценки полезно проверить, что API вообще доступно.
    request = urllib.request.Request(
        url=f"{api_url.rstrip('/')}/health",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"/health failed with HTTP {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach API at {api_url}: {error.reason}") from error


def ensure_api_ready(api_url: str) -> None:
    # Не запускаем оценку, если индекс пустой или Qdrant недоступен,
    # иначе метрики будут бессмысленны.
    health = get_health(api_url)
    status = str(health.get("status", "")).strip().lower()
    collection_name = str(health.get("collection_name", "unknown"))
    indexed_chunks = int(health.get("indexed_chunks", 0) or 0)

    if status != "ok":
        raise RuntimeError(
            "API is reachable, but its health status is degraded. "
            "Check that Qdrant is running and that backend .env points QDRANT_URL to the right host "
            "(default: http://localhost:6333)."
        )

    if indexed_chunks == 0:
        raise RuntimeError(
            f"API is reachable, but collection '{collection_name}' is empty. "
            "Ingest documents before running retrieval evaluation."
        )


def post_search(api_url: str, query: str, limit: int) -> list[dict]:
    # Скрипт меряет именно backend как черный ящик, поэтому идет через HTTP API,
    # а не вызывает Python-код напрямую.
    payload = json.dumps({"query": query, "limit": limit}).encode("utf-8")
    request = urllib.request.Request(
        url=f"{api_url.rstrip('/')}/search",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"/search failed with HTTP {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach API at {api_url}: {error.reason}") from error

    return body.get("hits", [])


def dedupe_paths(paths: list[Path]) -> list[Path]:
    # `/search` возвращает чанки, а метрики у нас считаются по документам,
    # поэтому объединяем повторяющиеся попадания из одного и того же файла.
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def reciprocal_rank(hit_paths: list[Path], relevant_paths: set[Path], max_rank: int) -> float:
    # MRR: чем раньше встретился первый релевантный документ, тем лучше.
    for rank, path in enumerate(hit_paths[:max_rank], start=1):
        if path in relevant_paths:
            return 1.0 / rank
    return 0.0


def evaluate_row(
    row: dict[str, str],
    catalog: dict[str, Path],
    api_url: str,
) -> dict[str, object]:
    expected_doc_ids = split_semicolon(row.get("expected_doc_ids", ""))
    missing_doc_ids = [doc_id for doc_id in expected_doc_ids if doc_id not in catalog]
    if missing_doc_ids:
        raise RuntimeError(
            f"Unknown doc_id(s) in eval row {row.get('query_id', '<unknown>')}: {', '.join(missing_doc_ids)}"
        )

    relevant_paths = {catalog[doc_id] for doc_id in expected_doc_ids}
    top_k = int(row.get("top_k", "5") or "5")
    # Retrieval returns chunks, so we request a wider pool and then collapse hits to unique documents.
    limit = max(25, top_k * 5)
    hits = post_search(api_url=api_url, query=row["query"], limit=limit)
    hit_paths = [Path(hit["source_path"]).resolve() for hit in hits]
    unique_hit_paths = dedupe_paths(hit_paths)

    relevant_in_5 = sum(1 for path in unique_hit_paths[:5] if path in relevant_paths)
    relevant_total = max(len(relevant_paths), 1)
    returned_top_5 = min(5, len(unique_hit_paths))

    recall_at_5 = relevant_in_5 / relevant_total
    precision_at_5 = relevant_in_5 / returned_top_5 if returned_top_5 else 0.0
    mrr_at_10 = reciprocal_rank(hit_paths=unique_hit_paths, relevant_paths=relevant_paths, max_rank=10)

    top_hits = ";".join(str(path) for path in unique_hit_paths[:5])
    matched_doc_ids = [
        doc_id for doc_id in expected_doc_ids if catalog[doc_id] in set(unique_hit_paths[:5])
    ]

    return {
        "query_id": row.get("query_id", ""),
        "query": row["query"],
        "expected_doc_ids": ";".join(expected_doc_ids),
        "matched_doc_ids_top5": ";".join(matched_doc_ids),
        "recall_at_5": f"{recall_at_5:.4f}",
        "precision_at_5": f"{precision_at_5:.4f}",
        "mrr_at_10": f"{mrr_at_10:.4f}",
        "top_5_hit_paths": top_hits,
    }


def write_report(output_path: Path, rows: list[dict[str, object]]) -> None:
    # Подробный CSV удобен для ручного разбора провальных запросов.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query",
        "expected_doc_ids",
        "matched_doc_ids_top5",
        "recall_at_5",
        "precision_at_5",
        "mrr_at_10",
        "top_5_hit_paths",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("No eval rows found.")
        return

    # Средние метрики по всем запросам дают быструю оценку качества ретривера.
    recall_at_5 = sum(float(row["recall_at_5"]) for row in rows) / len(rows)
    precision_at_5 = sum(float(row["precision_at_5"]) for row in rows) / len(rows)
    mrr_at_10 = sum(float(row["mrr_at_10"]) for row in rows) / len(rows)

    print(f"Queries evaluated: {len(rows)}")
    print(f"Recall@5: {recall_at_5:.4f}")
    print(f"Precision@5: {precision_at_5:.4f}")
    print(f"MRR@10: {mrr_at_10:.4f}")


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_path).resolve()
    catalog_path = Path(args.catalog_path).resolve()
    output_path = Path(args.output_path).resolve()

    # Последовательность простая: проверяем API -> загружаем датасет -> считаем метрики -> пишем отчет.
    ensure_api_ready(api_url=args.api_url)
    catalog = load_catalog(catalog_path)
    eval_rows = load_eval_rows(eval_path)
    report_rows = [evaluate_row(row=row, catalog=catalog, api_url=args.api_url) for row in eval_rows]
    write_report(output_path=output_path, rows=report_rows)
    print_summary(report_rows)
    print(f"Detailed report saved to: {output_path}")


if __name__ == "__main__":
    main()
