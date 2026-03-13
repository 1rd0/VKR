from groq import Groq

from app.models import SearchHit


class AnswerGenerator:
    def __init__(self, api_key: str | None, model_name: str, allow_fallback_answer: bool) -> None:
        self.model_name = model_name
        self.allow_fallback_answer = allow_fallback_answer
        self.client = Groq(api_key=api_key) if api_key else None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def build_context(self, hits: list[SearchHit]) -> str:
        fragments = []
        for index, hit in enumerate(hits, start=1):
            snippet = hit.text[:1200]
            fragments.append(f"[{index}] source={hit.source_path}\n{snippet}")
        return "\n\n".join(fragments)

    def answer(self, question: str, hits: list[SearchHit]) -> tuple[str, bool]:
        if not hits:
            return "В индексе пока нет релевантных фрагментов. Сначала загрузите документы.", False

        if not self.enabled:
            return self._fallback_answer(question, hits), False

        context = self.build_context(hits)
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты помощник для RAG-системы. Отвечай только на основе контекста. "
                    "Если в контексте не хватает данных, так и скажи. "
                    "В конце кратко перечисли источники, которые использовал."
                ),
            },
            {
                "role": "user",
                "content": f"Вопрос:\n{question}\n\nКонтекст:\n{context}",
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
            )
            content = response.choices[0].message.content or ""
            return content.strip(), True
        except Exception:
            return self._fallback_answer(question, hits), False

    def _fallback_answer(self, question: str, hits: list[SearchHit]) -> str:
        snippets = []
        for index, hit in enumerate(hits[:3], start=1):
            snippets.append(f"{index}. {hit.text[:280]} (source: {hit.source_path})")

        if not self.allow_fallback_answer:
            return "LLM недоступна, а fallback-ответ отключен."

        joined = "\n".join(snippets)
        return (
            f"LLM недоступна, поэтому ниже собраны самые близкие фрагменты по запросу '{question}'.\n"
            f"{joined}"
        )

