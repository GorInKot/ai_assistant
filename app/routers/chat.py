"""Эндпоинты диалога с ассистентом."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_db
from app.db import Conversation, Message, User
from app.schemas import AskAttachment, AskRequest, AskResponse, DialogClearRequest
from app.services.ask_service import clear_dialog, process_ask
from app.services.chat_documents import handle_chat_files
from app.services.document_processing import MAX_FILES, DocumentError
from app.state import llm_service


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


@router.post("/ask-files", response_model=AskResponse)
async def ask_files(
    message: str = Form(""),
    conversation_id: int | None = Form(None),
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AskResponse:
    """Обработка файлов, приложенных к сообщению в чате (Этап 6).

    Файлы читаются в память, обрабатываются и НЕ сохраняются на сервере.
    Сообщение и ответ пишутся в беседу (как обычный /ask)."""
    if not files:
        raise HTTPException(status_code=400, detail="Не приложено ни одного файла")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Слишком много файлов (лимит {MAX_FILES})")

    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        payloads.append((upload.filename or "file", await upload.read()))

    file_names = ", ".join(name for name, _ in payloads)
    user_text = message.strip()
    stored_user_content = (f"{user_text}\n\n" if user_text else "") + f"📎 {file_names}"

    conversation: Conversation | None = None
    if conversation_id is not None:
        conversation = _load_conversation(db, conversation_id, current_user)
        if conversation.title == DEFAULT_TITLE and not conversation.messages:
            conversation.title = _auto_title_from_question(user_text or file_names)
        db.add(Message(conversation_id=conversation.id, role="user", content=stored_user_content))
        conversation.updated_at = func.now()
        db.commit()

    try:
        result = handle_chat_files(user_text, payloads, llm_service)
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:  # LLM не настроен (выжимка)
        raise HTTPException(status_code=503, detail=str(exc))

    attachment = (
        AskAttachment(
            filename=result.attachment.filename,
            content_base64=result.attachment.content_base64,
            mime=result.attachment.mime,
        )
        if result.attachment
        else None
    )

    if conversation is not None:
        db.add(Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result.answer,
            sources_json="[]",
            no_exact_match=0,
        ))
        conversation.updated_at = func.now()
        db.commit()

    return AskResponse(
        answer=result.answer,
        sources=[],
        no_exact_match=False,
        conversation_id=conversation.id if conversation else None,
        attachment=attachment,
    )


@router.post("/dialog/clear")
def clear_dialog_state(
    payload: DialogClearRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    clear_dialog(payload.session_id, current_user.id)
    return {"status": "cleared"}
