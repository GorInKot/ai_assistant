from __future__ import annotations

import json
import re
from typing import Sequence

from openai import OpenAI

from app.kb import RetrievalResult


FALLBACK_RU = "В базе знаний нет точной информации по вашему вопросу"
FALLBACK_EN = "There is no exact information in the knowledge base for your question."


class LLMService:
    def __init__(self, api_key: str | None, model: str, enable_rerank: bool = True, rerank_candidates: int = 28) -> None:
        self.model = model
        self.enable_rerank = enable_rerank
        self.rerank_candidates = max(8, rerank_candidates)
        self.client = OpenAI(api_key=api_key) if api_key else None

    def rerank_results(self, question: str, candidates: Sequence[RetrievalResult], top_n: int = 16) -> list[RetrievalResult]:
        shortlist = list(candidates[: self.rerank_candidates])
        if not shortlist:
            return []

        if not self.client or not self.enable_rerank:
            return shortlist[:top_n]

        blocks: list[str] = []
        for idx, result in enumerate(shortlist, start=1):
            chunk = result.chunk
            location_parts = [chunk.relative_path]
            if chunk.page is not None:
                location_parts.append(f"page={chunk.page}")
            if chunk.section:
                location_parts.append(f"section={chunk.section}")
            location = " | ".join(location_parts)

            compact_text = " ".join(chunk.text.split())
            compact_text = compact_text[:420]
            blocks.append(f"{idx}. {location}\n{compact_text}")

        system_prompt = (
            "Ты модуль ранжирования корпоративного RAG. "
            "Твоя задача: выбрать фрагменты, которые помогут ответить на вопрос пользователя. "
            "Учитывай синонимы и перефразировки (например, участники=акторы=роли). "
            "Верни только JSON формата {\"selected_ids\": [..]} без пояснений. "
            "Выбери до 8 самых полезных id, начиная с наиболее релевантного."
        )

        user_prompt = (
            f"Вопрос:\n{question}\n\n"
            f"Кандидаты:\n" + "\n\n".join(blocks)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            payload = (response.choices[0].message.content or "").strip()
            selected_ids = self._parse_selected_ids(payload, max_id=len(shortlist), max_count=min(8, top_n))
        except Exception:
            return shortlist[:top_n]

        if not selected_ids:
            return shortlist[:top_n]

        selected_set = set(selected_ids)
        reranked = [shortlist[idx - 1] for idx in selected_ids if 1 <= idx <= len(shortlist)]
        for idx, result in enumerate(shortlist, start=1):
            if len(reranked) >= top_n:
                break
            if idx in selected_set:
                continue
            reranked.append(result)

        return reranked[:top_n]

    def generate_answer(self, question: str, context_results: Sequence[RetrievalResult], intent: str = "procedure") -> str:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not set")

        context_payload = self._format_context(context_results)
        language_hint = self._detect_language(question)

        brevity_hint = (
            "Если вопрос про определение системы, дай только определение и назначение (2-4 предложения), без шагов процесса."
            if intent == "definition"
            else "Если вопрос про документы, перечисли только релевантные документы и кратко поясни назначение каждого."
            if intent == "documents"
            else "Если вопрос про процесс, дай пошаговый ответ/чек-лист только по запрошенному процессу."
        )

        system_prompt = (
            "Ты корпоративный ассистент по внутренним процессам. "
            "Используй только информацию из блока КОНТЕКСТ. "
            "Если данных не хватает, ответь дословно: 'В базе знаний нет точной информации по вашему вопросу'. "
            "Учитывай, что пользователь может использовать синонимы терминов из документов. "
            "Не придумывай факты, номера документов, роли или шаги. "
            "Не добавляй детали, которые пользователь не запрашивал. "
            f"{brevity_hint} "
            f"Отвечай на языке запроса ({language_hint})."
        )

        user_prompt = (
            f"ВОПРОС:\n{question}\n\n"
            f"КОНТЕКСТ:\n{context_payload}\n\n"
            "Сформируй ответ только по контексту. "
            "Если есть неоднозначность, явно отметь это."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return (response.choices[0].message.content or "").strip()

    def _parse_selected_ids(self, payload: str, max_id: int, max_count: int) -> list[int]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return []

        selected = data.get("selected_ids")
        if not isinstance(selected, list):
            return []

        normalized: list[int] = []
        seen: set[int] = set()
        for item in selected:
            if not isinstance(item, int):
                continue
            if item < 1 or item > max_id:
                continue
            if item in seen:
                continue
            seen.add(item)
            normalized.append(item)
            if len(normalized) >= max_count:
                break

        return normalized

    def _format_context(self, context_results: Sequence[RetrievalResult]) -> str:
        blocks: list[str] = []
        for idx, result in enumerate(context_results, start=1):
            chunk = result.chunk
            location_parts = [chunk.relative_path]
            if chunk.page is not None:
                location_parts.append(f"page={chunk.page}")
            if chunk.section:
                location_parts.append(f"section={chunk.section}")
            location = " | ".join(location_parts)

            blocks.append(f"[{idx}] {location}\n{chunk.text}")

        return "\n\n".join(blocks)

    def _detect_language(self, text: str) -> str:
        en_count = len(re.findall(r"[A-Za-z]", text))
        ru_count = len(re.findall(r"[А-Яа-яЁё]", text))
        if en_count > max(ru_count * 1.2, 3):
            return "English"
        return "Russian"
