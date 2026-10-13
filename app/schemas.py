"""Pydantic-модели запросов/ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    # Если передан — сообщения сохраняются в эту беседу.
    # Если None — обратная совместимость со старым UI: ответ возвращается, но в БД не пишется.
    conversation_id: int | None = None


class AskAttachment(BaseModel):
    filename: str
    content_base64: str
    mime: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    no_exact_match: bool
    conversation_id: int | None = None
    # Файл-результат операции над документом (напр. объединённый xlsx) — для скачивания.
    attachment: AskAttachment | None = None
    # Проверка обоснованности ответа по найденным фрагментам: цитаты и неподтверждённые
    # утверждения (только для ответов, сгенерированных моделью). См. services/grounding.py.
    grounding: dict | None = None
    # Что за ответ: answer — по документам; missing — в базе не нашлось; clarify — просьба
    # уточнить вопрос; smalltalk — ответ об ассистенте; file — результат работы с файлом. Нужно журналу вопросов (services/gaps.py).
    kind: str = "answer"
    # id сохранённого сообщения ассистента — по нему интерфейс отправляет оценку ответа.
    message_id: int | None = None
    # Вопрос с учётом уточнения; только для журнала, в ответ API не попадает.
    resolved_question: str | None = Field(default=None, exclude=True)
    # Документы, из которых собран ответ (в интерфейсе показываются не все: markdown-заметки
    # скрыты), — для журнала вопросов. Если не задано, берутся показанные источники.
    log_sources: list[dict] | None = Field(default=None, exclude=True)


class RatingRequest(BaseModel):
    # 1 — полезно, -1 — не полезно, null — снять оценку.
    rating: int | None = Field(default=None)


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


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6, max_length=200)


class UserSummary(BaseModel):
    id: int
    email: str
    full_name: str | None
    division: str | None
    subdivision: str | None
    created_at: str | None
    roles: list[str]


class UserRolesUpdate(BaseModel):
    roles: list[str]


class AdminPasswordResetRequest(BaseModel):
    new_password: str = Field(..., min_length=6, max_length=200)
