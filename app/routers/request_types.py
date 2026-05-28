"""CRUD каталога типов заявок (admin/manager).

После любого CRUD сбрасываем in-memory кэш каталога — иначе ask_service
будет видеть устаревший список до рестарта.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_db, require_role
from app.db import (
    ROLE_ADMIN,
    ROLE_MANAGER,
    RequestType,
    RequestTypeSlot,
    ResponsibilityArea,
    User,
)
from app.request_catalog import reload_catalog
from app.schemas import RequestTypeIn, RequestTypeOut


admin_only = require_role(ROLE_ADMIN, ROLE_MANAGER)

router = APIRouter(prefix="/api/admin/request-types")


def _serialize(rt: RequestType) -> dict:
    try:
        triggers = json.loads(rt.trigger_keywords_json or "[]")
    except json.JSONDecodeError:
        triggers = []
    try:
        examples = json.loads(rt.examples_json or "[]")
    except json.JSONDecodeError:
        examples = []
    return {
        "id": rt.id,
        "type_slug": rt.type_slug,
        "title": rt.title,
        "responsibility_area_slug": rt.responsibility_area_slug,
        "is_anonymous": bool(rt.is_anonymous),
        "is_active": bool(rt.is_active),
        "trigger_keywords": [str(t) for t in triggers],
        "examples": [str(e) for e in examples],
        "sort_order": rt.sort_order,
        "slots": [
            {
                "id": s.id,
                "name": s.name,
                "question": s.question,
                "required": bool(s.required),
                "sort_order": s.sort_order,
            }
            for s in sorted(rt.slots, key=lambda x: x.sort_order)
        ],
    }


def _validate_area(db: Session, area_slug: str) -> None:
    if not db.query(ResponsibilityArea).filter(ResponsibilityArea.slug == area_slug).first():
        raise HTTPException(
            status_code=400,
            detail=f"Область ответственности '{area_slug}' не существует",
        )


def _replace_slots(db: Session, rt: RequestType, new_slots: list) -> None:
    for slot in list(rt.slots):
        db.delete(slot)
    db.flush()
    seen_names: set[str] = set()
    for order, item in enumerate(new_slots):
        if item.name in seen_names:
            raise HTTPException(
                status_code=400,
                detail=f"Имя слота '{item.name}' дублируется",
            )
        seen_names.add(item.name)
        db.add(RequestTypeSlot(
            request_type_id=rt.id,
            name=item.name,
            question=item.question,
            required=item.required,
            sort_order=order,
        ))


@router.get("", response_model=list[RequestTypeOut])
def list_request_types(
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> list[dict]:
    items = db.query(RequestType).order_by(RequestType.sort_order, RequestType.id).all()
    return [_serialize(rt) for rt in items]


@router.post("", response_model=RequestTypeOut, status_code=201)
def create_request_type(
    payload: RequestTypeIn,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict:
    if db.query(RequestType).filter(RequestType.type_slug == payload.type_slug).first():
        raise HTTPException(status_code=400, detail="Тип с таким slug уже существует")
    _validate_area(db, payload.responsibility_area_slug)

    max_order = db.query(RequestType).count()
    rt = RequestType(
        type_slug=payload.type_slug,
        title=payload.title,
        responsibility_area_slug=payload.responsibility_area_slug,
        is_anonymous=payload.is_anonymous,
        is_active=payload.is_active,
        trigger_keywords_json=json.dumps(payload.trigger_keywords, ensure_ascii=False),
        examples_json=json.dumps(payload.examples, ensure_ascii=False),
        sort_order=max_order,
    )
    db.add(rt)
    db.flush()
    _replace_slots(db, rt, payload.slots)
    db.commit()
    db.refresh(rt)
    reload_catalog()
    return _serialize(rt)


@router.get("/{type_slug}", response_model=RequestTypeOut)
def get_request_type_admin(
    type_slug: str,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict:
    rt = db.query(RequestType).filter(RequestType.type_slug == type_slug).first()
    if not rt:
        raise HTTPException(status_code=404, detail="Тип не найден")
    return _serialize(rt)


@router.put("/{type_slug}", response_model=RequestTypeOut)
def update_request_type(
    type_slug: str,
    payload: RequestTypeIn,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict:
    rt = db.query(RequestType).filter(RequestType.type_slug == type_slug).first()
    if not rt:
        raise HTTPException(status_code=404, detail="Тип не найден")

    if payload.type_slug != rt.type_slug:
        if db.query(RequestType).filter(RequestType.type_slug == payload.type_slug).first():
            raise HTTPException(status_code=400, detail="Slug уже занят другим типом")
        rt.type_slug = payload.type_slug

    _validate_area(db, payload.responsibility_area_slug)
    rt.title = payload.title
    rt.responsibility_area_slug = payload.responsibility_area_slug
    rt.is_anonymous = payload.is_anonymous
    rt.is_active = payload.is_active
    rt.trigger_keywords_json = json.dumps(payload.trigger_keywords, ensure_ascii=False)
    rt.examples_json = json.dumps(payload.examples, ensure_ascii=False)

    _replace_slots(db, rt, payload.slots)
    db.commit()
    db.refresh(rt)
    reload_catalog()
    return _serialize(rt)


@router.delete("/{type_slug}")
def delete_request_type(
    type_slug: str,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    rt = db.query(RequestType).filter(RequestType.type_slug == type_slug).first()
    if not rt:
        raise HTTPException(status_code=404, detail="Тип не найден")
    # Жёсткое удаление: каскадно удалит слоты. Уже созданные Request-ы
    # (с этим type_slug) остаются — у них type_slug хранится строкой,
    # без FK. Если нужно скрыть из новых заявок, не удаляя — используй is_active=false.
    db.delete(rt)
    db.commit()
    reload_catalog()
    return {"status": "deleted"}
