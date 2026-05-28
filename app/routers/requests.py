"""API заявок: список, детали, смена статуса.

Доступы:
- GET /api/requests/inbox — все заявки, назначенные на сотрудника, связанного
  с текущим юзером по Employee.user_id == User.id. Связь устанавливается
  ТОЛЬКО админом (через create/update employee либо явный link-endpoint),
  чтобы регистрация на чужой email не давала автоматический inbox.
- GET /api/requests/my — заявки, которые юзер сам создал (включая анонимные —
  юзер видит только то, что сам отправил).
- GET /api/requests/{id} — детали (только если автор или назначенный
  сотрудник по этой заявке).
- PUT /api/requests/{id}/status — смена статуса (только назначенный сотрудник
  или admin).

Анонимные обращения: автор скрыт от получателя (requester_name=None в API).
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_db, user_has_role
from app.db import (
    REQUEST_STATUSES,
    Employee,
    Request,
    RequestEvent,
    ROLE_ADMIN,
    User,
)
from app.schemas import RequestOut, RequestStatusUpdate


router = APIRouter(prefix="/api/requests")


def _employee_for_user(db: Session, user: User) -> Employee | None:
    """Связь User → Employee по Employee.user_id.

    Раньше использовалось совпадение email, но это позволяло перехватить чужой
    inbox через регистрацию на сотрудничский email (нет email-верификации).
    Теперь связь устанавливается ТОЛЬКО админом при создании/обновлении
    сотрудника либо через явный link-endpoint.
    """
    return db.query(Employee).filter(Employee.user_id == user.id).first()


def _serialize(request: Request, db: Session, hide_requester: bool) -> dict:
    requester_name: str | None = None
    requester_email: str | None = None
    if not hide_requester and request.requester_user_id:
        requester = db.query(User).filter(User.id == request.requester_user_id).first()
        if requester:
            requester_name = requester.full_name or requester.email
            requester_email = requester.email

    assigned_name: str | None = None
    if request.assigned_employee_id:
        emp = db.query(Employee).filter(Employee.id == request.assigned_employee_id).first()
        if emp:
            assigned_name = emp.full_name

    try:
        payload = json.loads(request.payload_json or "{}")
    except json.JSONDecodeError:
        payload = {}

    return {
        "id": request.id,
        "type_slug": request.type_slug,
        "type_title": request.type_title,
        "is_anonymous": request.is_anonymous,
        "status": request.status,
        "summary": request.summary,
        "payload": payload,
        "requester_name": requester_name,
        "requester_email": requester_email,
        "assigned_employee_id": request.assigned_employee_id,
        "assigned_employee_name": assigned_name,
        "conversation_id": request.conversation_id,
        "created_at": request.created_at.isoformat() if request.created_at else "",
        "updated_at": request.updated_at.isoformat() if request.updated_at else "",
    }


@router.get("/inbox", response_model=list[RequestOut])
def inbox(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    employee = _employee_for_user(db, current_user)
    if not employee:
        return []
    items = (
        db.query(Request)
        .filter(Request.assigned_employee_id == employee.id)
        .order_by(Request.id.desc())
        .all()
    )
    return [_serialize(r, db, hide_requester=r.is_anonymous) for r in items]


@router.get("/my", response_model=list[RequestOut])
def my_requests(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    items = (
        db.query(Request)
        .filter(Request.requester_user_id == current_user.id)
        .order_by(Request.id.desc())
        .all()
    )
    return [_serialize(r, db, hide_requester=False) for r in items]


@router.get("/{request_id}", response_model=RequestOut)
def get_request(
    request_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    request = db.query(Request).filter(Request.id == request_id).first()
    if not request:
        raise HTTPException(status_code=404, detail="Заявка не найдена")

    employee = _employee_for_user(db, current_user)
    is_recipient = employee is not None and request.assigned_employee_id == employee.id
    is_author = request.requester_user_id == current_user.id and not request.is_anonymous
    is_admin = user_has_role(current_user, ROLE_ADMIN)

    if not (is_recipient or is_author or is_admin):
        raise HTTPException(status_code=403, detail="Нет доступа к этой заявке")

    return _serialize(request, db, hide_requester=request.is_anonymous and not is_admin)


@router.put("/{request_id}/status", response_model=RequestOut)
def update_status(
    request_id: int,
    payload: RequestStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if payload.status not in REQUEST_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Статус должен быть одним из: {list(REQUEST_STATUSES)}",
        )

    request = db.query(Request).filter(Request.id == request_id).first()
    if not request:
        raise HTTPException(status_code=404, detail="Заявка не найдена")

    employee = _employee_for_user(db, current_user)
    is_recipient = employee is not None and request.assigned_employee_id == employee.id
    is_admin = user_has_role(current_user, ROLE_ADMIN)

    if not (is_recipient or is_admin):
        raise HTTPException(status_code=403, detail="Только получатель или admin может менять статус")

    request.status = payload.status
    db.add(RequestEvent(
        request_id=request.id,
        event_type=f"status:{payload.status}",
        actor_user_id=current_user.id,
        comment=payload.comment,
    ))
    db.commit()
    db.refresh(request)
    return _serialize(request, db, hide_requester=request.is_anonymous and not is_admin)
