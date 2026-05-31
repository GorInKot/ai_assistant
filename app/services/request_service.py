"""Создание Request в БД на основе PendingRequest + slot-filling helpers."""

from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from app.db import REQUEST_STATUS_NEW, Request, RequestEvent, User
from app.dialog_state import PendingRequest
from app.request_catalog import RequestTypeDef
from app.services.responsibilities import lookup_responsible


logger = logging.getLogger(__name__)


# Слова, которыми пользователь пропускает необязательный слот. «нет» НЕ входит:
# на этапе слота оно классифицируется как confirm_no и отменяет всю заявку.
SKIP_TOKENS = frozenset({
    "пропустить", "пропусти", "пропуск", "skip", "-", "—", "–",
    "далее", "дальше", "потом", "later", "n/a", "не знаю",
})


def is_skip_answer(text: str) -> bool:
    return text.strip().lower().strip(".!") in SKIP_TOKENS


def find_slot(type_def: RequestTypeDef, slot_name: str):
    for slot in type_def.slots:
        if slot.name == slot_name:
            return slot
    return None


def next_slot(pending: PendingRequest, type_def: RequestTypeDef) -> str | None:
    """Имя следующего слота для опроса (в порядке каталога).

    Спрашиваем как обязательные, так и необязательные слоты; пропущенные
    (skipped_slots) и уже заполненные — не повторяем. Возвращает None, когда
    каждый слот либо заполнен, либо пропущен. Поскольку обязательный слот
    пропустить нельзя, None гарантирует, что все required собраны.
    """
    for slot in type_def.slots:
        if slot.name in pending.filled_slots or slot.name in pending.skipped_slots:
            continue
        return slot.name
    return None


def next_required_slot(pending: PendingRequest, type_def: RequestTypeDef) -> str | None:
    """Имя следующего обязательного незаполненного слота (без учёта optional)."""
    for slot in type_def.slots:
        if slot.required and slot.name not in pending.filled_slots:
            return slot.name
    return None


def slot_question(type_def: RequestTypeDef, slot_name: str) -> str:
    slot = find_slot(type_def, slot_name)
    return slot.question if slot else f"Уточните: {slot_name}"


def slot_prompt(type_def: RequestTypeDef, slot_name: str) -> str:
    """Вопрос по слоту + подсказка про пропуск для необязательных полей."""
    slot = find_slot(type_def, slot_name)
    if slot is None:
        return f"Уточните: {slot_name}"
    if slot.required:
        return slot.question
    return f"{slot.question} (необязательно — можно ответить «пропустить»)"


def summarize_pending(type_def: RequestTypeDef, pending: PendingRequest) -> str:
    """Сводка перед подтверждением — показываем ТОЛЬКО заполненные поля.

    Раньше показывали все слоты включая незаполненные (с прочерком),
    что выглядело как «я тебя спросил, а ты не ответил» для тех слотов,
    которые ассистент вообще не задавал (required=false).
    """
    lines = [f"Тип заявки: {type_def.title}"]
    has_filled = False
    for slot in type_def.slots:
        value = pending.filled_slots.get(slot.name, "").strip()
        if not value:
            continue
        lines.append(f"• {slot.question.rstrip('?')} — {value}")
        has_filled = True
    if not has_filled:
        lines.append("(полей не указано)")
    lines.append("")
    lines.append("Подтверждаете создание заявки? Ответьте «да» или «нет».")
    return "\n".join(lines)


def finalize_request(
    db: Session,
    pending: PendingRequest,
    type_def: RequestTypeDef,
    current_user: User,
    conversation_id: int | None,
) -> tuple[Request, User | None]:
    """Сохраняет Request в БД, ищет ответственного, добавляет событие создания.

    Возвращает (request, assigned_employee). assigned_employee=None если по
    области нет назначенного ответственного — заявка всё равно сохраняется,
    но в UI будет видна с пометкой «не назначен».
    """
    assigned_employee = lookup_responsible(
        db,
        area_slug=type_def.responsibility_area,
        division=current_user.division,
        subdivision=current_user.subdivision,
    )

    summary = _build_summary(type_def, pending)

    request = Request(
        type_slug=type_def.type,
        type_title=type_def.title,
        requester_user_id=None if type_def.is_anonymous else current_user.id,
        assigned_employee_id=assigned_employee.id if assigned_employee else None,
        conversation_id=conversation_id,
        is_anonymous=type_def.is_anonymous,
        status=REQUEST_STATUS_NEW,
        payload_json=json.dumps(pending.filled_slots, ensure_ascii=False),
        summary=summary,
    )
    db.add(request)
    db.flush()

    db.add(RequestEvent(
        request_id=request.id,
        event_type="created",
        actor_user_id=None if type_def.is_anonymous else current_user.id,
        comment=None,
    ))
    db.commit()
    db.refresh(request)
    return request, assigned_employee


def _build_summary(type_def: RequestTypeDef, pending: PendingRequest) -> str:
    """Короткое summary для отображения в inbox-таблице."""
    if not pending.filled_slots:
        return type_def.title
    first_value = next(iter(pending.filled_slots.values()))
    return f"{type_def.title}: {first_value[:80]}"
