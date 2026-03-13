# Baseline RAG для ВКР

Это базовая версия RAG без модуля выявления и разрешения противоречий. Она нужна как контрольная точка:

- парсит PDF, HTML, TXT и Markdown;
- режет документы на чанки;
- считает эмбеддинги через `ai-forever/ru-en-RoSBERTa` c `CLS pooling` и retrieval-префиксами;
- складывает чанки в Qdrant;
- ищет релевантные фрагменты;
- формирует ответ через Groq + Llama либо возвращает fallback-ответ по найденным кускам.

## Окружение

Для текущего проекта используем обычный `venv + requirements.txt`. Это проще, предсказуемее и быстрее под локальную разработку, когда ты часто меняешь код.

## Структура

- `back/` - FastAPI API и RAG-логика;
- `front/` - Streamlit demo;
- `docker-compose.yml` - по умолчанию только Qdrant, а `api` и `demo` оставлены как optional `fullstack`-профиль;
- `back/data/raw/` - папка для документов.

## Быстрый старт для разработки

1. Скопируй настройки:

```bash
cp .env.example .env
```

2. Если хочешь генерацию ответа через LLM, пропиши `GROQ_API_KEY` в `.env`.

3. Подними только Qdrant:

```bash
docker compose up -d qdrant
```

4. Запусти backend локально:

```bash
cd back
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

5. В отдельном терминале запусти demo локально:

```bash
cd front
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

6. После старта открой:

- API: `http://localhost:8000/docs`
- Demo UI: `http://localhost:8501`
- Qdrant: `http://localhost:6333/dashboard`

На первом запросе локальный backend скачает retrieval-модель с Hugging Face в локальный кэш `back/.cache/huggingface`. Поэтому первый запуск может быть заметно дольше обычного, а следующие уже быстрее.

## Как загрузить документы

Есть два варианта:

1. Положить файлы в `back/data/raw/`, затем нажать `Index default directory` в demo.
2. Загрузить файлы прямо через Streamlit UI.

Для первого прогона уже добавлен пример: `back/data/raw/demo_policy.md`.

## Полный Docker-режим

Если позже захочешь снова поднять все в контейнерах:

```bash
docker compose --profile fullstack up --build
```

## Как скачивается `ru-en-RoSBERTa`

Модель не лежит в репозитории и не ставится прямо в git. Происходит так:

1. `pip install -r requirements.txt` ставит библиотеки `torch` и `transformers`.
2. Сам файл модели скачивается только при первом `ingest` или первом `search/ask`.
3. Скачивание запускается в [embeddings.py](/home/rabdel/projects/sVKR/back/app/services/embeddings.py#L1) на строках с `AutoTokenizer.from_pretrained(...)` и `AutoModel.from_pretrained(...)`.
4. После этого файлы остаются в локальном кэше `back/.cache/huggingface`, и повторно качаться не должны.

Если хочешь скачать модель заранее, без запуска UI:

```bash
cd back
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "from app.services.embeddings import TextEncoder; TextEncoder('ai-forever/ru-en-RoSBERTa')"
```

## Базовый pipeline

1. Документ парсится в текст.
2. Текст режется на чанки с overlap.
3. Каждый чанк кодируется retrieval-моделью.
4. Вектор и метаданные пишутся в Qdrant.
5. На вопрос пользователя ищутся top-k чанков.
6. Эти чанки подаются в LLM для финального ответа.

## Что это дает для ВКР

Эта версия подходит как baseline, с которым потом можно сравнить:

- качество retrieval;
- качество final answer;
- поведение на разных версиях документов;
- эффект от будущего contradiction-модуля.

## Baseline evaluation

Для baseline-версии заведен отдельный набор метрик в [evaluation/baseline_rag_v1/README.md](/home/rabdel/projects/sVKR/evaluation/baseline_rag_v1/README.md).

Шаблон для фиксации результатов лежит в [evaluation/baseline_rag_v1/metrics_template.csv](/home/rabdel/projects/sVKR/evaluation/baseline_rag_v1/metrics_template.csv).

Это удобно для ВКР: baseline не размазывается по заметкам, а оформлен как отдельная контрольная точка, с которой потом можно честно сравнивать улучшения.

## Что делать дальше после baseline

Следующий логичный шаг:

1. Добавить явные поля `document_type`, `document_version`, `valid_from`, `valid_to`.
2. Разметить небольшой evaluation-набор в `Pandas`.
3. Поверх retrieval внедрить contradiction classification через `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
