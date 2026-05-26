"""Эндпоинты диалога с ассистентом."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth import get_current_user
from app.db import User
from app.schemas import AskRequest, AskResponse, DialogClearRequest
from app.services.ask_service import clear_dialog, process_ask


router = APIRouter(prefix="/api")


@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    current_user: User = Depends(get_current_user),
) -> AskResponse:
    return process_ask(payload.question, payload.session_id, current_user.id)


@router.post("/dialog/clear")
def clear_dialog_state(
    payload: DialogClearRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    clear_dialog(payload.session_id, current_user.id)
    return {"status": "cleared"}
