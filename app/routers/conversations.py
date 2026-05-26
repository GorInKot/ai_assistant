"""CRUD-эндпоинты бесед пользователя (ChatGPT-подобный сайдбар)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_db
from app.db import Conversation, Message, User
from app.schemas import ConversationCreateRequest, ConversationPatchRequest


router = APIRouter(prefix="/api/conversations")


def _serialize_conversation(conv: Conversation) -> dict:
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }


def _serialize_message(msg: Message) -> dict:
    sources: list[dict] = []
    if msg.sources_json:
        try:
            parsed = json.loads(msg.sources_json)
            if isinstance(parsed, list):
                sources = parsed
        except json.JSONDecodeError:
            sources = []
    return {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "sources": sources,
        "no_exact_match": bool(msg.no_exact_match),
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }


def _get_user_conversation(db: Session, conversation_id: int, user: User) -> Conversation:
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Беседа не найдена")
    return conv


@router.get("")
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, list[dict]]:
    items = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return {"conversations": [_serialize_conversation(item) for item in items]}


@router.post("")
def create_conversation(
    payload: ConversationCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    title = (payload.title or "").strip() or "Новая беседа"
    conv = Conversation(user_id=current_user.id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return _serialize_conversation(conv)


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    conv = _get_user_conversation(db, conversation_id, current_user)
    return {
        **_serialize_conversation(conv),
        "messages": [_serialize_message(msg) for msg in conv.messages],
    }


@router.patch("/{conversation_id}")
def rename_conversation(
    conversation_id: int,
    payload: ConversationPatchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    conv = _get_user_conversation(db, conversation_id, current_user)
    conv.title = payload.title.strip()
    conv.updated_at = func.now()
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return _serialize_conversation(conv)


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    conv = _get_user_conversation(db, conversation_id, current_user)
    db.delete(conv)
    db.commit()
    return {"status": "deleted"}
