"""Локальный журнал действий (заявок/обращений) MVP."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.auth import get_current_user
from app.db import User
from app.schemas import ActionCreateRequest
from app.state import actions_store


router = APIRouter(prefix="/api")


@router.get("/actions")
def list_actions(
    limit: int = Query(default=50, ge=1, le=500),
    block: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
) -> dict[str, list[dict]]:
    return {"actions": actions_store.list_actions(limit=limit, block=block)}


@router.post("/actions")
def create_action(
    payload: ActionCreateRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, object]:
    requester_value = payload.requester.strip() or (current_user.full_name or current_user.email)

    record = actions_store.create_action(
        action_type=payload.action_type.strip(),
        process=payload.process.strip(),
        block=payload.block.strip() if payload.block else None,
        status=payload.status.strip() if payload.status else "Черновик",
        title=payload.title.strip(),
        details=payload.details.strip(),
        requester=requester_value,
    )
    return {
        "status": "created",
        "message": "Действие зарегистрировано локально (MVP, без интеграции с корпоративными ИС).",
        "action": {
            "action_id": record.action_id,
            "action_type": record.action_type,
            "process": record.process,
            "title": record.title,
            "details": record.details,
            "requester": record.requester,
            "status": record.status,
            "created_at": record.created_at,
        },
    }
