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


class EmployeeIn(BaseModel):
    email: str = Field(..., min_length=5, max_length=160)
    full_name: str = Field(..., min_length=2, max_length=200)
    position: str | None = Field(default=None, max_length=200)
    division: str | None = Field(default=None, max_length=120)
    subdivision: str | None = Field(default=None, max_length=120)
    phone: str | None = Field(default=None, max_length=40)
    is_active: bool = True
    # slug-и областей ответственности, которые сотрудник курирует
    responsibility_area_slugs: list[str] = Field(default_factory=list)


class EmployeeOut(BaseModel):
    id: int
    user_id: int | None
    email: str
    full_name: str
    position: str | None
    division: str | None
    subdivision: str | None
    phone: str | None
    is_active: bool
    responsibility_area_slugs: list[str]

    model_config = {"from_attributes": True}


class ResponsibilityAreaIn(BaseModel):
    slug: str = Field(..., min_length=2, max_length=80, pattern=r"^[a-z0-9_-]+$")
    name: str = Field(..., min_length=2, max_length=200)
    description: str | None = Field(default=None, max_length=500)


class ResponsibilityAreaOut(BaseModel):
    id: int
    slug: str
    name: str
    description: str | None

    model_config = {"from_attributes": True}


class ResponsibilityAssign(BaseModel):
    employee_id: int
    area_slug: str
    scope_division: str | None = None
    scope_subdivision: str | None = None
    is_primary: bool = True


class ResponsibilityOut(BaseModel):
    id: int
    employee_id: int
    employee_full_name: str
    employee_email: str
    area_slug: str
    area_name: str
    scope_division: str | None
    scope_subdivision: str | None
    is_primary: bool


class EmployeeImportResult(BaseModel):
    created: int
    updated: int
    skipped: int
    errors: list[str]


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


class RequestOut(BaseModel):
    id: int
    type_slug: str
    type_title: str
    is_anonymous: bool
    status: str
    summary: str | None
    payload: dict
    requester_name: str | None
    requester_email: str | None
    assigned_employee_id: int | None
    assigned_employee_name: str | None
    conversation_id: int | None
    created_at: str
    updated_at: str


class RequestStatusUpdate(BaseModel):
    status: str = Field(..., min_length=2, max_length=40)
    comment: str | None = Field(default=None, max_length=2000)


class RequestTypeSlotIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    question: str = Field(..., min_length=2, max_length=400)
    required: bool = False


class RequestTypeSlotOut(RequestTypeSlotIn):
    id: int
    sort_order: int


class RequestTypeIn(BaseModel):
    type_slug: str = Field(..., min_length=2, max_length=80, pattern=r"^[a-z0-9_]+$")
    title: str = Field(..., min_length=2, max_length=200)
    responsibility_area_slug: str = Field(..., min_length=2, max_length=80)
    is_anonymous: bool = False
    is_active: bool = True
    trigger_keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    slots: list[RequestTypeSlotIn] = Field(default_factory=list)


class RequestTypeOut(BaseModel):
    id: int
    type_slug: str
    title: str
    responsibility_area_slug: str
    is_anonymous: bool
    is_active: bool
    trigger_keywords: list[str]
    examples: list[str]
    sort_order: int
    slots: list[RequestTypeSlotOut]
