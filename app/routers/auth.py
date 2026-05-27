"""Регистрация, логин и профиль текущего пользователя по токену."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import (
    assign_initial_role,
    create_access_token,
    get_current_user,
    get_db,
    get_password_hash,
    verify_password,
)
from app.db import User
from app.schemas import LoginRequest, RegisterRequest, TokenResponse


router = APIRouter(prefix="/api")


@router.post("/auth/register", response_model=TokenResponse)
def register_user(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Пароли не совпадают")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Пользователь с таким email уже существует")

    full_name = " ".join(
        part
        for part in (
            payload.last_name.strip(),
            payload.first_name.strip(),
            (payload.middle_name or "").strip(),
        )
        if part
    )
    user = User(
        email=payload.email,
        full_name=full_name,
        first_name=payload.first_name.strip(),
        last_name=payload.last_name.strip(),
        middle_name=(payload.middle_name or "").strip() or None,
        division=payload.division.strip(),
        subdivision=(payload.subdivision or "").strip() or None,
        hashed_password=get_password_hash(payload.password),
    )
    assign_initial_role(db, user)
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)


@router.post("/auth/login", response_model=TokenResponse)
def login_user(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)


@router.get("/user/profile")
def get_current_user_profile(current_user: User = Depends(get_current_user)) -> dict[str, object]:
    return {
        "email": current_user.email,
        "full_name": current_user.full_name,
        "created_at": current_user.created_at,
        "roles": [role.name for role in current_user.roles],
    }
