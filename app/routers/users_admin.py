"""Управление пользователями и ролями (только admin).

Здесь admin может:
- видеть всех пользователей и их роли,
- менять набор ролей,
- сбрасывать пароль (новый назначается без знания старого).

Защита от случайного снятия admin-роли с самого себя: при попытке убрать
admin у себя возвращаем 400 — это предотвращает блокировку системы.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_db, get_password_hash, require_role
from app.db import ROLE_ADMIN, Role, User
from app.schemas import AdminPasswordResetRequest, UserRolesUpdate, UserSummary


admin_only = require_role(ROLE_ADMIN)

router = APIRouter(prefix="/api/admin/users")


def _serialize_user(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "division": user.division,
        "subdivision": user.subdivision,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "roles": [role.name for role in user.roles],
    }


@router.get("", response_model=list[UserSummary])
def list_users(
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> list[dict]:
    items = db.query(User).order_by(User.id).all()
    return [_serialize_user(u) for u in items]


@router.put("/{user_id}/roles", response_model=UserSummary)
def update_user_roles(
    user_id: int,
    payload: UserRolesUpdate,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    # Защита: нельзя снять admin у самого себя — иначе можно случайно
    # лишиться доступа ко всей админке и в одиночку не вернуть его обратно.
    if user.id == current_user.id and ROLE_ADMIN not in payload.roles:
        raise HTTPException(
            status_code=400,
            detail="Нельзя забрать admin-роль у самого себя",
        )

    available_roles = {r.name: r for r in db.query(Role).all()}
    unknown = [name for name in payload.roles if name not in available_roles]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Неизвестные роли: {unknown}")

    user.roles = [available_roles[name] for name in payload.roles]
    db.commit()
    db.refresh(user)
    return _serialize_user(user)


@router.post("/{user_id}/password", status_code=200)
def reset_password(
    user_id: int,
    payload: AdminPasswordResetRequest,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    user.hashed_password = get_password_hash(payload.new_password)
    db.commit()
    return {"status": "password_reset"}


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Нельзя удалить самого себя")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    db.delete(user)
    db.commit()
    return {"status": "deleted"}
