# Baseline Evaluation

В этой папке лежат шаблоны для оценки baseline RAG.

## Файлы

- `eval_retrieval.csv` - вопросы для retrieval-оценки.
- `eval_answer.csv` - вопросы для answer-оценки.
- `results/` - сюда скрипт складывает отчеты по retrieval.

## Как заполнять `eval_retrieval.csv`

Обязательные поля:

- `query_id` - уникальный идентификатор вопроса.
- `query` - вопрос пользователя.
- `expected_doc_ids` - один или несколько `doc_id` из `back/data/catalog.csv`, разделенные `;`.
- `top_k` - сколько чанков запрашивать у `/search`.

Дополнительные поля:

- `expected_versions` - ожидаемые версии через `;`.
- `notes` - комментарий по смыслу вопроса.

Пример:

```csv
query_id,query,expected_doc_ids,expected_versions,top_k,notes
rag_001,Что такое Копилка для сдачи?,kbo-2024-roundup;kbo-2025-roundup,2024-09-16;2025-02-03,2,Определение услуги
```

## Как заполнять `eval_answer.csv`

Этот файл пока нужен как шаблон для ручной или полуавтоматической оценки answer-качества.

Обязательные поля:

- `question_id`
- `question`
- `expected_doc_ids`
- `reference_answer`
- `expected_facts`
- `forbidden_facts`
- `top_k`

`expected_facts` и `forbidden_facts` разделяй через `;`.

## Как считать retrieval-метрики

Из директории `back/`:

```bash
source .venv/bin/activate
python scripts/evaluate_baseline.py
```

Скрипт:

- читает `eval/eval_retrieval.csv`;
- читает `data/catalog.csv`;
- дергает локальный `POST /search`;
- дедуплицирует несколько чанков одного и того же файла по `source_path`;
- считает `Recall@5`, `Precision@5`, `MRR@10`;
- сохраняет детальный отчет в `eval/results/retrieval_report.csv`.

Если API поднят не на `127.0.0.1:8000`, передай адрес явно:

```bash
python scripts/evaluate_baseline.py --api-url http://127.0.0.1:8000
```
