# Baseline Metrics: `baseline_rag_v1`

Это отдельная контрольная точка для базового RAG-пайплайна без модуля выявления и разрешения противоречий.

Идея простая: все первые замеры качества складываются сюда, а следующие эксперименты сравниваются именно с этим baseline.

## Что измерять

### Retrieval

- `Recall@5`
- `Recall@10`
- `MRR@5`
- `nDCG@10`

### Answer quality

- `answer_correctness`
- `answer_faithfulness`
- `answer_completeness`

### System

- `latency_search_ms`
- `latency_answer_ms`
- `context_tokens`

## Как хранить результаты

Заполняй файл `metrics_template.csv` построчно для каждого запуска.

Рекомендуемый `run_id`:

- `baseline_rag_v1_dev_001`
- `baseline_rag_v1_dev_002`
- `baseline_rag_v1_test_001`

## Правило сравнения

Все будущие версии (`reranker`, `contradiction module`, новые embedding-модели, новые prompt-стратегии) сравниваются с этим baseline по тем же вопросам и на том же evaluation-наборе.
