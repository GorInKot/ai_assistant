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


def next_required_slot(pending: PendingRequest, type_def: RequestTypeDef) -> str | None:
    """Имя следующего обязательного слота, который ещё не заполнен."""
    for slot in type_def.slots:
        if slot.required and slot.name not in pending.filled_slots:
            return slot.name
    return None


def slot_question(type_def: RequestTypeDef, slot_name: str) -> str:
    for slot in type_def.slots:
        if slot.name == slot_name:
            return slot.question
    return f"Уточните: {slot_name}"


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
