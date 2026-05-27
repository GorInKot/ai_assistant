"""Профиль пользователя (ЛК): данные о подразделении, ФИО, должности."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_db, get_password_hash, verify_password
from app.db import User
from app.profile import ALLOWED_DIVISIONS, BRANCH_SUBDIVISIONS, UserProfile
from app.schemas import PasswordChangeRequest, ProfileRequest


router = APIRouter(prefix="/api")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _user_to_profile_dict(user: User) -> dict[str, str]:
    subdivision = user.subdivision or ""
    return {
        "full_name": user.full_name or "",
        "division": user.division or "",
        "subdivision": subdivision,
        # Старый ключ для обратной совместимости с UI-кэшем.
        "subdivision_type": subdivision,
        "job_title": user.job_title or "",
        "email": user.email,
    }


def _normalize_profile_payload(payload: ProfileRequest) -> UserProfile:
    full_name = re.sub(r"\s+", " ", payload.full_name).strip()
    division = payload.division.strip()
    subdivision = (payload.subdivision or payload.subdivision_type or "").strip()
    job_title = re.sub(r"\s+", " ", payload.job_title).strip()
    email = payload.email.strip().lower()

    if division not in ALLOWED_DIVISIONS:
        raise HTTPException(status_code=400, detail="Некорректное подразделение.")

    if division == "ЦА":
        subdivision = ""
    else:
        allowed_subdivisions = BRANCH_SUBDIVISIONS.get(division, ())
        if subdivision not in allowed_subdivisions:
            raise HTTPException(
                status_code=400,
                detail=f"Для '{division}' укажите корректное ПУ/АУП.",
            )

    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Некорректный email.")

    return UserProfile(
        full_name=full_name,
        division=division,
        subdivision=subdivision or None,
        job_title=job_title,
        email=email,
    )


@router.get("/profile")
def get_profile(current_user: User = Depends(get_current_user)) -> dict[str, object]:
    return {
        "profile": _user_to_profile_dict(current_user),
        "options": {
            "divisions": list(ALLOWED_DIVISIONS),
            "subdivisions_by_division": {
                division: list(subdivisions)
                for division, subdivisions in BRANCH_SUBDIVISIONS.items()
            },
        },
    }


@router.post("/profile")
def save_profile(
    payload: ProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    profile = _normalize_profile_payload(payload)

    # email менять нельзя — он используется как идентификатор пользователя в JWT.
    current_user.full_name = profile.full_name
    current_user.division = profile.division
    current_user.subdivision = profile.subdivision
    current_user.job_title = profile.job_title
    db.add(current_user)
    db.commit()
    db.refresh(current_user)

    return {"status": "saved", "profile": _user_to_profile_dict(current_user)}


@router.post("/profile/password")
def change_password(
    payload: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Текущий пароль неверный")
    current_user.hashed_password = get_password_hash(payload.new_password)
    db.add(current_user)
    db.commit()
    return {"status": "changed"}
