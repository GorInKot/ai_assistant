"""Эндпоинты диалога с ассистентом."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_db
from app.db import Conversation, Message, User
from app.schemas import AskRequest, AskResponse, DialogClearRequest
from app.services.ask_service import clear_dialog, process_ask


router = APIRouter(prefix="/api")

DEFAULT_TITLE = "Новая беседа"
AUTO_TITLE_MAX_LEN = 60


def _auto_title_from_question(question: str) -> str:
    cleaned = " ".join(question.split())
    if len(cleaned) <= AUTO_TITLE_MAX_LEN:
        return cleaned or DEFAULT_TITLE
    return cleaned[: AUTO_TITLE_MAX_LEN - 1].rstrip() + "…"


def _load_conversation(db: Session, conversation_id: int, user: User) -> Conversation:
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Беседа не найдена")
    return conv


@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AskResponse:
    raw_question = payload.question.strip()
    conversation: Conversation | None = None

    # Если фронт передал conversation_id — пишем историю в БД.
    # Без conversation_id остаётся обратная совместимость со старым UI: история не пишется.
    if payload.conversation_id is not None:
        conversation = _load_conversation(db, payload.conversation_id, current_user)

        # Auto-title: если это первое сообщение и заголовок дефолтный, ставим начало вопроса.
        if conversation.title == DEFAULT_TITLE and not conversation.messages:
            conversation.title = _auto_title_from_question(raw_question)

        db.add(Message(
            conversation_id=conversation.id,
            role="user",
            content=raw_question,
        ))
        conversation.updated_at = func.now()
        db.commit()

    response = process_ask(
        raw_question=payload.question,
        session_id_raw=payload.session_id,
        current_user=current_user,
        db=db,
        conversation_id=conversation.id if conversation else None,
    )

    if conversation is not None:
        db.add(Message(
            conversation_id=conversation.id,
            role="assistant",
            content=response.answer,
            sources_json=json.dumps(response.sources, ensure_ascii=False),
            no_exact_match=1 if response.no_exact_match else 0,
        ))
        conversation.updated_at = func.now()
        db.commit()
        response.conversation_id = conversation.id

    return response


@router.post("/dialog/clear")
def clear_dialog_state(
    payload: DialogClearRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    clear_dialog(payload.session_id, current_user.id)
    return {"status": "cleared"}
