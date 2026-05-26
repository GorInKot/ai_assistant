"""Pydantic-модели запросов/ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    # Если передан — сообщения сохраняются в эту беседу.
    # Если None — обратная совместимость со старым UI: ответ возвращается, но в БД не пишется.
    conversation_id: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    no_exact_match: bool
    conversation_id: int | None = None


class DialogClearRequest(BaseModel):
    session_id: str | None = None


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationPatchRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ActionCreateRequest(BaseModel):
    action_type: str = Field(..., min_length=2)
    process: str = Field(..., min_length=2)
    block: str | None = Field(default=None, min_length=2)
    status: str = Field(default="Черновик", min_length=2)
    title: str = Field(..., min_length=2)
    details: str = Field(..., min_length=2)
    requester: str = Field(default="")


class ProfileRequest(BaseModel):
    full_name: str = Field(..., min_length=3, max_length=120)
    division: str = Field(..., min_length=2, max_length=64)
    subdivision: str | None = Field(default=None, max_length=64)
    subdivision_type: str | None = Field(default=None, max_length=64)
    job_title: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., min_length=5, max_length=160)


class RegisterRequest(BaseModel):
    email: str
    password: str
    confirm_password: str
    last_name: str
    first_name: str
    middle_name: str | None = None
    division: str
    subdivision: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
