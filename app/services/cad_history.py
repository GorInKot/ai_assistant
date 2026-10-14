"""История запросов конструктора деталей: запись, список, удаление (на пользователя)."""

from __future__ import annotations

import json
from datetime import timezone

from sqlalchemy.orm import Session

from app.db import CadRequest

OUTCOME_BUILT = "built"
OUTCOME_CLARIFY = "clarify"
OUTCOME_UNSUPPORTED = "unsupported"
OUTCOME_ERROR = "error"

MAX_PER_USER = 200  # старые записи удаляются, чтобы история не росла бесконечно
DEFAULT_LIMIT = 50


def record(
    db: Session,
    user_id: int,
    description: str,
    outcome: str,
    *,
    title: str | None = None,
    detail: str | None = None,
    spec: dict | None = None,
    notes: list[str] | None = None,
) -> CadRequest | None:
    """Записать запрос. Ошибка записи не должна ломать ответ пользователю."""
    try:
        row = CadRequest(
            user_id=user_id,
            description=description.strip(),
            outcome=outcome,
            title=title,
            detail=detail,
            spec_json=json.dumps(spec, ensure_ascii=False) if spec else None,
            notes_json=json.dumps(notes, ensure_ascii=False) if notes else None,
        )
        db.add(row)
        db.commit()
        old = (
            db.query(CadRequest.id)
            .filter(CadRequest.user_id == user_id)
            .order_by(CadRequest.id.desc())
            .offset(MAX_PER_USER)
            .all()
        )
        if old:
            db.query(CadRequest).filter(CadRequest.id.in_([r.id for r in old])).delete(synchronize_session=False)
            db.commit()
        return row
    except Exception:  # noqa: BLE001 — история вторична
        db.rollback()
        return None


def serialize(row: CadRequest) -> dict:
    created = row.created_at
    if created is not None and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)  # SQLite хранит UTC без пометки пояса
    return {
        "id": row.id,
        "created_at": created.isoformat() if created else None,
        "description": row.description,
        "outcome": row.outcome,
        "title": row.title,
        "detail": row.detail,
        "can_rebuild": row.outcome == OUTCOME_BUILT and bool(row.spec_json),
    }


def list_for_user(db: Session, user_id: int, limit: int = DEFAULT_LIMIT) -> list[dict]:
    rows = (
        db.query(CadRequest)
        .filter(CadRequest.user_id == user_id)
        .order_by(CadRequest.id.desc())
        .limit(limit)
        .all()
    )
    return [serialize(r) for r in rows]


def get_own(db: Session, user_id: int, request_id: int) -> CadRequest | None:
    return db.query(CadRequest).filter(CadRequest.id == request_id, CadRequest.user_id == user_id).first()


def stored_spec(row: CadRequest) -> tuple[dict, list[str]]:
    """Сохранённая спецификация и пометки записи."""
    return json.loads(row.spec_json or "{}"), json.loads(row.notes_json or "[]")
