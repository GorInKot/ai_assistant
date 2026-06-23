from __future__ import annotations

import json
import re
from typing import Sequence

from openai import OpenAI

from app.kb import RetrievalResult


FALLBACK_RU = "В базе знаний нет точной информации по вашему вопросу"
FALLBACK_EN = "There is no exact information in the knowledge base for your question."


class LLMService:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        enable_rerank: bool = True,
        rerank_candidates: int = 28,
    ) -> None:
        self.model = model
        self.enable_rerank = enable_rerank
        self.rerank_candidates = max(8, rerank_candidates)

        if base_url:
            # OpenAI-совместимый бэкенд (Groq, Ollama, LM Studio, vLLM).
            # Локальным серверам ключ обычно не нужен — подставляем заглушку.
            self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        elif api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None


    def classify_intent(
        self,
        message: str,
        request_types_summary: list[dict],
    ) -> dict:
        """LLM-классификация входящего сообщения.

        Возвращает JSON-словарь:
            {"intent": "qa" | "create_request" | "confirm_yes" | "confirm_no" | "cancel",
             "request_type": "<slug или null>"}

        Если клиент не настроен или вызов упал — возвращаем "qa" (fallback на обычный QA).
        Это безопасный дефолт: пользователь не теряет возможность получить ответ.
        """
        fallback = {"intent": "qa", "request_type": None}

        if not self.client:
            return fallback

        catalog_lines = []
        for rt in request_types_summary:
            kw = ", ".join(rt.get("trigger_keywords", []))
            examples = "; ".join(rt.get("examples", [])[:2])
            catalog_lines.append(
                f'- "{rt["type"]}" ({rt["title"]}): triggers=[{kw}]; examples: {examples}'
            )
        catalog_block = "\n".join(catalog_lines) if catalog_lines else "(каталог пуст)"

        system_prompt = (
            "Ты классификатор намерений в корпоративном чате. "
            "На вход — сообщение пользователя. "
            "Определи intent. Возможные значения:\n"
            "  qa — обычный вопрос по базе знаний (по умолчанию).\n"
            "  create_request — пользователь хочет ОФОРМИТЬ заявку. "
            "Выбери подходящий request_type из каталога.\n"
            "  confirm_yes — пользователь подтверждает текущее действие "
            "(коротко: 'да', 'подтверждаю', 'давай', 'ок').\n"
            "  confirm_no — пользователь отказывается ('нет', 'не надо', 'отмена').\n"
            "  cancel — пользователь явно хочет отменить процесс ('отмени', 'забудь').\n\n"
            "Каталог типов заявок:\n"
            f"{catalog_block}\n\n"
            'Верни СТРОГО JSON: {"intent": "...", "request_type": "<slug|null>"}. '
            "Никаких пояснений вне JSON."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            data = json.loads(content)
        except (Exception,):
            return fallback

        intent = str(data.get("intent", "qa"))
        if intent not in {"qa", "create_request", "confirm_yes", "confirm_no", "cancel"}:
            intent = "qa"
        request_type = data.get("request_type")
        if request_type is not None:
            request_type = str(request_type)
        return {"intent": intent, "request_type": request_type}

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
            raise RuntimeError("LLM не настроен: задайте LLM_BASE_URL и/или LLM_API_KEY")

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

    @staticmethod
    def _summary_length_instruction(text: str) -> str:
        """Целевая длина выжимки пропорционально объёму (≈10% со «скобками»).

        Правило из RoadMap (Этап 6, #6): короткий текст не раздуваем, длинный —
        не ужимаем в пару фраз. Диапазоны по числу слов входа.
        """
        n_words = len(text.split())
        if n_words < 300:
            return "Это короткий текст — уложись в 2-3 предложения."
        if n_words < 2000:
            return "Сделай резюме одним абзацем, примерно 100-150 слов."
        if n_words < 10000:
            return (
                "Сделай структурированное резюме на 200-300 слов: "
                "1-2 вводных предложения, затем ключевые пункты списком."
            )
        return (
            "Сделай развёрнутое структурированное резюме на 300-400 слов "
            "с разбивкой на тематические пункты списком."
        )

    def summarize_text(self, text: str) -> str:
        """Краткая выжимка большого текста (6.B.2 роадмапа).

        Переиспользует тот же LLM-клиент, что и QA. Длина резюме подбирается
        пропорционально объёму текста (см. `_summary_length_instruction`). Язык
        резюме — язык исходного текста. Если клиент не настроен — понятная ошибка
        (RuntimeError), роутер превратит её в 503.
        """
        if not self.client:
            raise RuntimeError("LLM не настроен: задайте LLM_BASE_URL и/или LLM_API_KEY")

        language_hint = self._detect_language(text)
        system_prompt = (
            "Ты делаешь краткое резюме документа. "
            "Сохрани ключевые факты, решения и сроки; убери воду. "
            "Не придумывай факты, которых нет в тексте. "
            f"{self._summary_length_instruction(text)} "
            f"Отвечай на языке документа ({language_hint})."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"ТЕКСТ ДОКУМЕНТА:\n{text}"},
                ],
            )
        except Exception as exc:  # noqa: BLE001 — сетевые/auth-ошибки LLM-провайдера
            raise RuntimeError(
                "Не удалось получить выжимку: модель недоступна "
                "(проверьте LLM_API_KEY/доступ к провайдеру)."
            ) from exc
        return (response.choices[0].message.content or "").strip()

    def extract_fields(self, text: str, fields: list[dict]) -> dict:
        """Извлечь значения полей из свободного текста пользователя (Этап 7).

        `fields` — список словарей {key, title, description}. Возвращает
        {key: значение|null}. Если клиент не настроен или вызов упал —
        понятная ошибка (RuntimeError), роутер превратит её в 503.
        """
        if not self.client:
            raise RuntimeError("LLM не настроен: задайте LLM_BASE_URL и/или LLM_API_KEY")

        field_lines = "\n".join(
            f'- "{f["key"]}": {f["title"]}'
            + (f" — {f['description']}" if f.get("description") else "")
            for f in fields
        )
        keys = ", ".join(f'"{f["key"]}"' for f in fields)
        system_prompt = (
            "Ты извлекаешь структурированные данные из свободного текста для "
            "заполнения заявки. Верни СТРОГО JSON-объект только с ключами: "
            f"{keys}. "
            "Если значение поля в тексте не указано — поставь null. "
            "Не придумывай данные, которых нет в тексте. "
            "Сохраняй значения как в тексте (ФИО, должности, телефоны, ключи без изменений).\n\n"
            f"Поля:\n{field_lines}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            data = json.loads(content)
        except Exception as exc:  # noqa: BLE001 — сетевые/auth/JSON-ошибки
            raise RuntimeError(
                "Не удалось распознать данные заявки: модель недоступна "
                "(проверьте LLM_API_KEY/доступ к провайдеру)."
            ) from exc

        if not isinstance(data, dict):
            return {f["key"]: None for f in fields}

        result: dict[str, str | None] = {}
        for f in fields:
            value = data.get(f["key"])
            if value is None:
                result[f["key"]] = None
            else:
                text_value = str(value).strip()
                result[f["key"]] = text_value or None
        return result

    def extract_people(self, text: str, fields: list[dict]) -> list[dict]:
        """Извлечь СПИСОК пользователей из свободного текста (Этап 7, групповые заявки).

        В отличие от `extract_fields` (один человек), коллективные заявки (ВКД)
        перечисляют нескольких сотрудников. Возвращает список словарей
        {key: значение|null} — по одному на каждого распознанного пользователя.
        Порядок сохраняется. Если ничего не распознано — пустой список.
        """
        if not self.client:
            raise RuntimeError("LLM не настроен: задайте LLM_BASE_URL и/или LLM_API_KEY")

        field_lines = "\n".join(
            f'- "{f["key"]}": {f["title"]}'
            + (f" — {f['description']}" if f.get("description") else "")
            for f in fields
        )
        keys = ", ".join(f'"{f["key"]}"' for f in fields)
        system_prompt = (
            "Ты извлекаешь данные о НЕСКОЛЬКИХ пользователях из свободного текста "
            "для заполнения групповой заявки. Верни СТРОГО JSON-объект вида "
            '{"users": [ {объект на пользователя}, ... ]}. '
            f"Каждый объект пользователя содержит только ключи: {keys}. "
            "Если значение поля для пользователя не указано — поставь null. "
            "Не придумывай данные, которых нет в тексте. Если в тексте один "
            "пользователь — верни список из одного объекта. Сохраняй значения "
            "как в тексте (ФИО, e-mail, учётные записи, ключи без изменений).\n\n"
            f"Поля каждого пользователя:\n{field_lines}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            data = json.loads(content)
        except Exception as exc:  # noqa: BLE001 — сетевые/auth/JSON-ошибки
            raise RuntimeError(
                "Не удалось распознать данные заявки: модель недоступна "
                "(проверьте LLM_API_KEY/доступ к провайдеру)."
            ) from exc

        raw_users = data.get("users") if isinstance(data, dict) else None
        if not isinstance(raw_users, list):
            return []

        people: list[dict] = []
        for item in raw_users:
            if not isinstance(item, dict):
                continue
            person: dict[str, str | None] = {}
            for f in fields:
                value = item.get(f["key"])
                if value is None:
                    person[f["key"]] = None
                else:
                    text_value = str(value).strip()
                    person[f["key"]] = text_value or None
            # Пропускаем полностью пустые объекты.
            if any(person.values()):
                people.append(person)
        return people

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
