"""Бизнес-логика обработки вопросов к ассистенту.

Все хелперы поиска источников, формирования fallback-ответов
и определения участников живут здесь. Роутер /api/ask должен
оставаться тонким и вызывать только process_ask().
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from fastapi import HTTPException

from sqlalchemy.orm import Session

from app.db import User
from app.dialog_state import PendingRequest
from app.kb import RetrievalResult
from app.llm import FALLBACK_EN, FALLBACK_RU
from app.request_catalog import RequestTypeDef, all_request_types, get_request_type
from app.roles import (
    ACTION_ROLE_WEIGHTS,
    ROLE_PATTERNS,
    extract_action_hints,
    is_participant_question,
    is_responsibility_question,
)
from app.schemas import AskResponse
from app.services.request_service import (
    finalize_request,
    find_slot,
    is_skip_answer,
    next_slot,
    slot_prompt,
    slot_question,
    summarize_pending,
)
from app.state import (
    dialog_state,
    kb_index,
    kb_lock,
    llm_service,
    request_logger,
    settings,
)


GENERIC_SOURCE_QUERY_TOKENS = {
    "систем",
    "ис",
    "инструкц",
    "документ",
    "документа",
    "файл",
    "процесс",
    "процес",
    "порядок",
    "вопрос",
    "информац",
    "ответ",
}

SEMANTIC_QUERY_EXPANSIONS: dict[str, set[str]] = {
    "ретрансляц": {"повторн", "отправк", "телематич", "данных"},
    "глонасс": {"телематич", "данных", "пробег"},
    "доступ": {"заявк", "форм"},
}


def detect_language(question: str) -> str:
    en_count = len(re.findall(r"[A-Za-z]", question))
    ru_count = len(re.findall(r"[А-Яа-яЁё]", question))
    if en_count > max(ru_count * 1.2, 3):
        return "en"
    return "ru"


def normalize_session_id(session_id: str | None, user_id: int) -> str:
    value = (session_id or "").strip()
    if not value:
        value = "default"
    if len(value) > 96:
        value = value[:96]
    # Префикс из user_id защищает от подделки чужой сессии диалога другим пользователем.
    return f"u{user_id}:{value}"


def extract_participants_from_results(
    results: list[RetrievalResult],
    process_hint: str | None = None,
    action_hints: set[str] | None = None,
    min_relative_score: float = 0.55,
    max_items: int = 10,
) -> list[str]:
    score_by_label: Counter[str] = Counter()

    for rank, result in enumerate(results):
        chunk = result.chunk
        if process_hint and chunk.process != process_hint:
            continue

        rank_bonus = max(0.2, 1.6 - (rank * 0.06))
        chunk_weight = (result.score * 0.35) + (result.coverage * 2.2) + rank_bonus
        chunk_text_lower = chunk.text.lower()

        if action_hints:
            action_hits = sum(1 for stem in action_hints if stem in chunk_text_lower)
            if action_hits == 0:
                chunk_weight *= 0.35
            else:
                chunk_weight *= 1.0 + min(0.4, action_hits * 0.12)

        for label, pattern in ROLE_PATTERNS:
            if pattern.search(chunk.text):
                label_weight = 1.0
                if action_hints:
                    for stem in action_hints:
                        role_weights = ACTION_ROLE_WEIGHTS.get(stem)
                        if not role_weights:
                            continue
                        label_weight *= role_weights.get(label, 0.95)
                label_weight = min(max(label_weight, 0.35), 2.8)
                score_by_label[label] += chunk_weight * label_weight

    ordered = score_by_label.most_common(max_items)
    if not ordered:
        return []

    top_score = ordered[0][1]
    min_score = max(1.2, top_score * min_relative_score)
    filtered = [label for label, score in ordered if score >= min_score]
    return filtered[:max_items]


def build_participants_answer(language: str, participants: list[str], process_hint: str | None) -> str:
    subject = process_hint.replace("_", " ") if process_hint else "указанном процессе"
    if language == "en":
        lines = "\n".join(f"- {item}" for item in participants)
        return f"Found process participant roles in {subject}:\n{lines}"

    lines = "\n".join(f"- {item}" for item in participants)
    return f"В базе знаний найдены участники (роли) заявочного процесса в {subject}:\n{lines}"


def build_responsibility_answer(language: str, participants: list[str], process_hint: str | None) -> str:
    subject = process_hint.replace("_", " ") if process_hint else "указанном процессе"
    top = participants[:2]
    if language == "en":
        lines = "\n".join(f"- {item}" for item in top)
        return f"According to the knowledge base, for {subject} this is handled by:\n{lines}"

    lines = "\n".join(f"- {item}" for item in top)
    if len(top) == 1:
        return f"По базе знаний в {subject} это выполняет:\n{lines}"
    return f"По базе знаний в {subject} это обычно выполняют:\n{lines}"


def build_fallback_answer(language: str, related_docs: list[dict], reason: str = "missing") -> str:
    ambiguous_suffix_en = " Please specify the process or system first."
    ambiguous_suffix_ru = " Уточните, пожалуйста, процесс или систему (например: ЕКТП, ЦУС, обучение/медосмотр)."

    if language == "en":
        prefix = FALLBACK_EN if reason == "missing" else "The request is ambiguous across multiple processes."
        if related_docs:
            return (
                f"{prefix}\n\n"
                "Please clarify the request (process, role, system, or document name), "
                "or review the related documents from the source list."
                + (ambiguous_suffix_en if reason == "ambiguous" else "")
            )
        return (
            f"{prefix}\n\n"
            "Please clarify the request (process, role, system, or document name)."
            + (ambiguous_suffix_en if reason == "ambiguous" else "")
        )

    prefix = FALLBACK_RU if reason == "missing" else "Запрос неоднозначен и может относиться к нескольким процессам."
    if related_docs:
        return (
            f"{prefix}\n\n"
            "Уточните вопрос (процесс, роль, систему или название документа) "
            "или откройте релевантные документы из блока источников."
            + (ambiguous_suffix_ru if reason == "ambiguous" else "")
        )

    return (
        f"{prefix}\n\n"
        "Уточните вопрос (процесс, роль, систему или название документа)."
        + (ambiguous_suffix_ru if reason == "ambiguous" else "")
    )


def _source_extension(source: dict) -> str:
    return Path(str(source.get("relative_path", ""))).suffix.lower()


def _source_process(source: dict) -> str:
    relative_path = str(source.get("relative_path", ""))
    return relative_path.split("/", 1)[0] if "/" in relative_path else "Общее"


def _dominant_context_process(context_results: list[RetrievalResult]) -> str | None:
    if not context_results:
        return None

    counts: Counter[str] = Counter(item.chunk.process for item in context_results)
    top_process, top_count = counts.most_common(1)[0]
    total = sum(counts.values())
    share = top_count / max(total, 1)

    if len(counts) == 1:
        return top_process
    if top_count >= 2 and share >= 0.6:
        return top_process
    return None


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for source in sources:
        key = str(source.get("relative_path", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


def _semantic_query_tokens(question: str, process_hint: str | None) -> set[str]:
    tokens = set(kb_index.query_tokens(question))
    process_tokens = kb_index.process_token_set(process_hint)
    result: set[str] = set()
    for token in tokens:
        if token.startswith("syn:"):
            continue
        if token in process_tokens:
            continue
        if token in GENERIC_SOURCE_QUERY_TOKENS:
            continue
        if len(token) < 3:
            continue
        result.add(token)

    expanded: set[str] = set()
    for token in result:
        for stem, additions in SEMANTIC_QUERY_EXPANSIONS.items():
            if token.startswith(stem):
                expanded.update(additions)
    result.update(expanded)

    return result


def _rank_related_sources_by_question(
    question: str,
    process_hint: str | None,
    related_sources: list[dict],
) -> list[dict]:
    semantic_tokens = _semantic_query_tokens(question, process_hint)
    if not semantic_tokens:
        return []

    scored: list[tuple[float, int, dict]] = []
    for source in related_sources:
        relative_path = str(source.get("relative_path", ""))
        record = kb_index.documents.get(relative_path)
        if not record:
            continue

        overlap = semantic_tokens & record.metadata_tokens
        if not overlap:
            continue

        score = sum(kb_index.token_idf.get(token, 1.0) for token in overlap)
        overlap_count = len(overlap)
        if overlap_count < 1:
            continue
        if overlap_count == 1 and score < 0.75:
            continue

        scored.append((score, overlap_count, source))

    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return [item[2] for item in scored]


def _role_query_process_fallback_sources(question: str, process_hint: str | None) -> list[dict]:
    if not process_hint:
        return []

    docs = kb_index.list_documents(query=None, process=process_hint, forms_only=False)
    non_md_docs = [doc for doc in docs if Path(str(doc.get("relative_path", ""))).suffix.lower() != ".md"]
    if not non_md_docs:
        return []

    q = question.lower()
    asks_access = "доступ" in q
    semantic_tokens = _semantic_query_tokens(question, process_hint)

    def is_instruction(doc: dict) -> bool:
        name = str(doc.get("file_name", "")).lower()
        return ("инструкц" in name) or ("руководств" in name) or ("manual" in name)

    def is_form(doc: dict) -> bool:
        return bool(doc.get("is_form"))

    scored: list[tuple[float, dict]] = []
    for doc in non_md_docs:
        path = str(doc.get("relative_path", ""))
        record = kb_index.documents.get(path)
        metadata_tokens = record.metadata_tokens if record else set()
        name_lower = str(doc.get("file_name", "")).lower()

        score = 0.0
        if is_instruction(doc):
            score += 2.0
        if not is_form(doc):
            score += 1.2
        if is_form(doc) and asks_access:
            score += 3.0
        if is_form(doc) and not asks_access:
            score -= 1.2

        overlap = semantic_tokens & metadata_tokens
        if overlap:
            score += 2.4 + (0.6 * len(overlap))

        if ("телемат" in name_lower or "ретрансля" in name_lower or "повторн" in name_lower) and not (
            "телемат" in q or "ретрансля" in q or "повторн" in q or "глонасс" in q
        ):
            score -= 2.6

        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_limit = 1 if not asks_access else 2

    output: list[dict] = []
    for _, doc in scored[:top_limit]:
        output.append(
            {
                "file_name": doc["file_name"],
                "relative_path": doc["relative_path"],
                "url": doc["url"],
                "download_url": doc["download_url"],
                "pages": [],
                "sections": [],
                "source_type": "related",
            }
        )
    return output


def _build_source_from_path(relative_path: str, source_type: str) -> dict | None:
    record = kb_index.documents.get(relative_path)
    if not record:
        return None

    return {
        "file_name": record.file_name,
        "relative_path": record.relative_path,
        "url": kb_index._build_file_url(record.relative_path),
        "download_url": kb_index._build_file_url(record.relative_path, download=True),
        "pages": [],
        "sections": [],
        "source_type": source_type,
    }


def _rank_sources_from_results(
    question: str,
    process_hint: str | None,
    results: list[RetrievalResult],
    source_type: str,
    exclude_paths: set[str] | None = None,
    require_metadata_overlap: bool = False,
    require_semantic_overlap: bool = False,
) -> list[dict]:
    semantic_tokens = _semantic_query_tokens(question, process_hint)
    exclude_paths = exclude_paths or set()

    score_by_path: dict[str, float] = {}
    pages_by_path: dict[str, set[int]] = {}
    sections_by_path: dict[str, set[str]] = {}

    for result in results:
        chunk = result.chunk
        relative_path = chunk.relative_path
        if relative_path in exclude_paths:
            continue
        if Path(relative_path).suffix.lower() == ".md":
            continue
        if process_hint and chunk.process != process_hint:
            continue

        record = kb_index.documents.get(relative_path)
        if not record:
            continue

        metadata_overlap: set[str] = set()
        if semantic_tokens:
            metadata_overlap = semantic_tokens & record.metadata_tokens
            if require_metadata_overlap and not metadata_overlap:
                continue

        score = (result.score * 0.9) + (result.coverage * 2.4)
        if semantic_tokens:
            chunk_tokens = kb_index.query_tokens(chunk.text)
            overlap = semantic_tokens & chunk_tokens
            if require_semantic_overlap and not overlap:
                continue
            if overlap:
                score += 1.0 + (0.4 * len(overlap))
            else:
                score *= 0.08
            if metadata_overlap:
                score += 1.2 + (0.45 * len(metadata_overlap))

        if chunk.page is not None:
            score += 0.2

        previous = score_by_path.get(relative_path, 0.0)
        score_by_path[relative_path] = max(previous, score)

        if chunk.page is not None:
            pages_by_path.setdefault(relative_path, set()).add(chunk.page)
        if chunk.section:
            sections_by_path.setdefault(relative_path, set()).add(chunk.section)

    ranked_paths = sorted(score_by_path.items(), key=lambda item: item[1], reverse=True)
    output: list[dict] = []
    for relative_path, score in ranked_paths:
        if semantic_tokens and score < 1.25:
            continue

        source = _build_source_from_path(relative_path, source_type=source_type)
        if not source:
            continue
        source["pages"] = sorted(pages_by_path.get(relative_path, set()))
        source["sections"] = sorted(sections_by_path.get(relative_path, set()))
        output.append(source)

    return output


def select_display_sources(
    question: str,
    process_hint: str | None,
    ranked_results: list[RetrievalResult],
    context_results: list[RetrievalResult],
    context_sources: list[dict],
    related_sources: list[dict],
) -> list[dict]:
    intent = kb_index.classify_query_intent(question)
    if is_participant_question(question) or is_responsibility_question(question):
        intent = "procedure"
    target_process = process_hint or _dominant_context_process(context_results)

    context_non_md = _rank_sources_from_results(
        question=question,
        process_hint=target_process,
        results=context_results,
        source_type="context",
        require_metadata_overlap=False,
        require_semantic_overlap=(intent != "documents"),
    )

    if context_non_md and intent != "documents":
        return context_non_md[:4]

    if (is_participant_question(question) or is_responsibility_question(question)) and not context_non_md:
        role_context_fallback = _rank_sources_from_results(
            question=question,
            process_hint=target_process,
            results=context_results,
            source_type="context",
            require_metadata_overlap=False,
            require_semantic_overlap=False,
        )
        if role_context_fallback:
            return _dedupe_sources(role_context_fallback)[:2]

    context_paths = {str(source.get("relative_path", "")) for source in context_non_md}
    ranked_from_retrieval = _rank_sources_from_results(
        question=question,
        process_hint=target_process,
        results=ranked_results,
        source_type="related",
        exclude_paths=context_paths,
        require_metadata_overlap=(intent != "documents"),
        require_semantic_overlap=(intent != "documents"),
    )

    if context_non_md and intent == "documents":
        combined = _dedupe_sources([*context_non_md, *ranked_from_retrieval])
        return combined[:6]

    if ranked_from_retrieval:
        return _dedupe_sources(ranked_from_retrieval)[:4]

    related_non_md = [source for source in related_sources if _source_extension(source) != ".md"]
    if target_process:
        process_related = [
            source for source in related_non_md if _source_process(source) == target_process
        ]
        if process_related:
            related_non_md = process_related

    ranked_related = _rank_related_sources_by_question(question, target_process, related_non_md)
    if ranked_related:
        return _dedupe_sources(ranked_related)[:4]

    if is_participant_question(question) or is_responsibility_question(question):
        role_process_fallback = _role_query_process_fallback_sources(question, target_process)
        if role_process_fallback:
            return role_process_fallback

    if intent != "documents":
        return []

    process_for_fallback = target_process or process_hint
    if process_for_fallback:
        process_docs = kb_index.list_documents(query=question, process=process_for_fallback, forms_only=False)
        fallback_docs = [
            {
                "file_name": doc["file_name"],
                "relative_path": doc["relative_path"],
                "url": doc["url"],
                "download_url": doc["download_url"],
                "pages": [],
                "sections": [],
                "source_type": "related",
            }
            for doc in process_docs
            if Path(str(doc["relative_path"])).suffix.lower() != ".md"
        ]
        return _dedupe_sources(fallback_docs)[:4]

    return []


def _build_catalog_summary() -> list[dict]:
    return [
        {
            "type": t.type,
            "title": t.title,
            "trigger_keywords": list(t.trigger_keywords),
            "examples": list(t.examples),
        }
        for t in all_request_types()
    ]


def _start_request_flow(
    session_id: str, type_def: RequestTypeDef, raw_question: str
) -> AskResponse:
    """Начинаем slot-filling для нового типа заявки."""
    slot_name = next_slot(PendingRequest(request_type=type_def.type), type_def)
    if slot_name is None:
        # У типа нет слотов — сразу confirmation.
        pending = PendingRequest(
            request_type=type_def.type,
            awaiting_confirmation=True,
        )
        dialog_state.set_pending_request(session_id, pending)
        return AskResponse(
            answer=summarize_pending(type_def, pending),
            sources=[],
            no_exact_match=False,
        )

    pending = PendingRequest(
        request_type=type_def.type,
        awaiting_slot=slot_name,
    )
    dialog_state.set_pending_request(session_id, pending)
    return AskResponse(
        answer=f"Хорошо, оформим заявку «{type_def.title}». {slot_prompt(type_def, slot_name)}",
        sources=[],
        no_exact_match=False,
    )


def _continue_slot_filling(
    session_id: str,
    pending: PendingRequest,
    type_def: RequestTypeDef,
    user_answer: str,
) -> AskResponse:
    """Обрабатываем ответ пользователя на текущий слот, переходим к следующему."""
    assert pending.awaiting_slot is not None
    current = pending.awaiting_slot
    slot = find_slot(type_def, current)

    if is_skip_answer(user_answer):
        if slot is not None and slot.required:
            # Обязательное поле пропустить нельзя — переспрашиваем, слот не меняем.
            dialog_state.set_pending_request(session_id, pending)
            return AskResponse(
                answer=f"Это поле обязательно, его нельзя пропустить. {slot_question(type_def, current)}",
                sources=[],
                no_exact_match=False,
            )
        pending.skipped_slots.add(current)
    else:
        pending.filled_slots[current] = user_answer.strip()

    pending.awaiting_slot = None

    slot_name = next_slot(pending, type_def)
    if slot_name is not None:
        pending.awaiting_slot = slot_name
        dialog_state.set_pending_request(session_id, pending)
        return AskResponse(
            answer=slot_prompt(type_def, slot_name),
            sources=[],
            no_exact_match=False,
        )

    # Все слоты собраны (required заполнены, optional заполнены/пропущены) — confirmation.
    pending.awaiting_confirmation = True
    dialog_state.set_pending_request(session_id, pending)
    return AskResponse(
        answer=summarize_pending(type_def, pending),
        sources=[],
        no_exact_match=False,
    )


def _finalize_and_respond(
    db: Session,
    session_id: str,
    pending: PendingRequest,
    type_def: RequestTypeDef,
    current_user: User,
    conversation_id: int | None,
) -> AskResponse:
    request, employee = finalize_request(
        db=db,
        pending=pending,
        type_def=type_def,
        current_user=current_user,
        conversation_id=conversation_id,
    )
    dialog_state.clear_pending_request(session_id)

    if employee:
        if type_def.is_anonymous:
            msg = (
                f"✅ Заявка #{request.id} «{type_def.title}» создана и направлена "
                f"ответственному ({employee.full_name}). Анонимность сохранена."
            )
        else:
            msg = (
                f"✅ Заявка #{request.id} «{type_def.title}» создана и направлена "
                f"ответственному: {employee.full_name} ({employee.email})."
            )
    else:
        msg = (
            f"⚠️ Заявка #{request.id} «{type_def.title}» создана, но по области "
            f"«{type_def.responsibility_area}» пока не назначен ответственный. "
            "Сообщите администратору — он добавит сотрудника."
        )

    return AskResponse(answer=msg, sources=[], no_exact_match=False)


def process_ask(
    raw_question: str,
    session_id_raw: str | None,
    current_user: User,
    db: Session,
    conversation_id: int | None = None,
) -> AskResponse:
    """Главная точка входа для /api/ask.

    Логика:
      1. Если есть pending_request (заявка в процессе) — обрабатываем как
         ответ на слот / подтверждение / отмену.
      2. Иначе — LLM-классификация интента. Если create_request — стартуем flow.
      3. Иначе — обычный QA с RAG.
    """
    raw_question = raw_question.strip()
    if not raw_question:
        raise HTTPException(status_code=400, detail="Question is empty")

    session_id = normalize_session_id(session_id_raw, current_user.id)

    # ---- 1. Pending request flow ----
    pending = dialog_state.get_pending_request(session_id)
    if pending is not None:
        type_def = get_request_type(pending.request_type)
        if type_def is None:
            # Каталог поменялся — сбрасываем pending.
            dialog_state.clear_pending_request(session_id)
        else:
            # Пропуск необязательного слота обрабатываем ДО intent-классификации:
            # реальный LLM нередко принимает «пропустить» за cancel и отменяет
            # всю заявку. Пока заполняется слот, skip имеет приоритет.
            if pending.awaiting_slot is not None and is_skip_answer(raw_question):
                response = _continue_slot_filling(session_id, pending, type_def, raw_question)
                request_logger.log(raw_question, [], answer=response.answer)
                return response

            catalog_summary = _build_catalog_summary()
            intent_data = llm_service.classify_intent(raw_question, catalog_summary)
            intent = intent_data["intent"]

            if intent == "cancel" or intent == "confirm_no":
                dialog_state.clear_pending_request(session_id)
                answer = (
                    "Хорошо, заявка отменена."
                    if pending.awaiting_confirmation
                    else "Хорошо, оформление заявки прервано."
                )
                request_logger.log(raw_question, [], answer=answer)
                return AskResponse(answer=answer, sources=[], no_exact_match=False)

            if pending.awaiting_confirmation:
                if intent == "confirm_yes":
                    response = _finalize_and_respond(
                        db, session_id, pending, type_def, current_user, conversation_id
                    )
                    request_logger.log(raw_question, [], answer=response.answer)
                    return response
                # Неоднозначный ответ — переспросим.
                answer = "Не понял ответ. Подтвердить заявку? Ответьте «да» или «нет»."
                request_logger.log(raw_question, [], answer=answer)
                return AskResponse(answer=answer, sources=[], no_exact_match=False)

            # awaiting_slot — пишем ответ пользователя в слот.
            if pending.awaiting_slot is not None:
                response = _continue_slot_filling(session_id, pending, type_def, raw_question)
                request_logger.log(raw_question, [], answer=response.answer)
                return response

    # ---- 2. Уточнение к незавершённому QA-вопросу приоритетнее классификации
    # интента. Иначе короткий уточняющий ответ (например «по ЕКТП» после
    # «кто участники процесса?») будет распознан intent-классификатором как
    # намерение создать заявку и перехватит диалог. Поэтому сначала пробуем
    # смержить с pending-уточнением, и только для действительно нового сообщения
    # запускаем intent-классификацию.
    effective_question, is_clarification = dialog_state.merge_with_pending(session_id, raw_question)

    if not is_clarification:
        catalog_summary = _build_catalog_summary()
        intent_data = llm_service.classify_intent(raw_question, catalog_summary)
        if intent_data["intent"] == "create_request":
            type_slug = intent_data.get("request_type")
            type_def = get_request_type(type_slug) if type_slug else None
            if type_def is not None:
                response = _start_request_flow(session_id, type_def, raw_question)
                request_logger.log(raw_question, [], answer=response.answer)
                return response

    # ---- 3. Обычный QA flow ----
    language = detect_language(effective_question)

    with kb_lock:
        process_hint = kb_index.detect_process_hint(effective_question)
        # Hybrid: BM25 + векторный поиск (RRF). Если эмбеддинги не настроены —
        # внутри отдаст чистый BM25.
        lexical_results = kb_index.hybrid_retrieve(
            effective_question,
            top_k=max(50, settings.rerank_candidates * 2),
        )
        reranked_results = llm_service.rerank_results(
            effective_question,
            lexical_results,
            top_n=settings.rerank_top_n,
        )

        intent = kb_index.classify_query_intent(effective_question)
        context_results = kb_index.select_context_results(effective_question, reranked_results, max_chunks=5)
        has_strong_context = kb_index.is_context_strong(context_results)
        process_ambiguous = kb_index.is_process_ambiguous(context_results)
        context_sources = kb_index.build_sources_from_results(context_results)

        exclude_paths = {item["relative_path"] for item in context_sources}
        related_sources = kb_index.find_related_documents(
            effective_question,
            limit=4,
            exclude_paths=exclude_paths,
        )
        display_sources = select_display_sources(
            question=effective_question,
            process_hint=process_hint,
            ranked_results=reranked_results,
            context_results=context_results,
            context_sources=context_sources,
            related_sources=related_sources,
        )

    final_sources = display_sources

    participant_query = is_participant_question(effective_question)
    responsibility_query = is_responsibility_question(effective_question)
    if participant_query or responsibility_query:
        # Без явно названного процесса вопрос про участников/ответственных
        # неоднозначен: роли у ЕКТП, ЦУС, обучения и т.д. разные. Не угадываем
        # по «самому громкому» процессу из выдачи, а просим уточнить — после
        # уточнения («по ЕКТП») process_hint определится и мы ответим точно.
        if process_hint is None:
            answer = build_fallback_answer(language, final_sources, reason="ambiguous")
            dialog_state.set_pending(session_id, effective_question)
            request_logger.log(raw_question, final_sources, answer=answer)
            return AskResponse(answer=answer, sources=final_sources, no_exact_match=True)

        # process_hint здесь гарантированно задан (иначе вышли бы выше).
        role_scope_process = process_hint
        action_hints = extract_action_hints(effective_question)
        single_owner_hints = {"доступ", "справоч", "акцепт"}

        if responsibility_query and not action_hints and not participant_query:
            answer = build_fallback_answer(language, final_sources, reason="ambiguous")
            dialog_state.set_pending(session_id, effective_question)
            request_logger.log(raw_question, final_sources, answer=answer)
            return AskResponse(answer=answer, sources=final_sources, no_exact_match=True)
        else:
            participants = extract_participants_from_results(
                reranked_results[: min(len(reranked_results), 24)],
                process_hint=role_scope_process,
                action_hints=action_hints,
                min_relative_score=0.72 if responsibility_query else 0.45,
            )
        min_items = 1 if responsibility_query else 2
        if len(participants) >= min_items:
            if responsibility_query and (action_hints & single_owner_hints):
                participants = participants[:1]
            if responsibility_query:
                answer = build_responsibility_answer(language, participants, process_hint=role_scope_process)
            else:
                answer = build_participants_answer(language, participants, process_hint=role_scope_process)
            dialog_state.clear(session_id)
            request_logger.log(raw_question, final_sources, answer=answer)
            return AskResponse(answer=answer, sources=final_sources, no_exact_match=False)

    if not has_strong_context or (process_ambiguous and process_hint is None):
        reason = "ambiguous" if process_ambiguous and process_hint is None else "missing"
        answer = build_fallback_answer(language, final_sources, reason=reason)
        if reason == "ambiguous":
            dialog_state.set_pending(session_id, effective_question)
        request_logger.log(raw_question, final_sources, answer=answer)
        return AskResponse(answer=answer, sources=final_sources, no_exact_match=True)

    try:
        answer = llm_service.generate_answer(
            effective_question,
            context_results,
            intent=intent,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    no_exact_match = FALLBACK_RU.lower() in answer.lower() or FALLBACK_EN.lower() in answer.lower()
    dialog_state.clear(session_id)
    request_logger.log(raw_question, final_sources, answer=answer)
    return AskResponse(answer=answer, sources=final_sources, no_exact_match=no_exact_match)


def clear_dialog(session_id_raw: str | None, user_id: int) -> None:
    session_id = normalize_session_id(session_id_raw, user_id)
    dialog_state.clear(session_id)
