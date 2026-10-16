"""Вход из корпоративного портала: ассистент открывается разделом портала без второго пароля.

Портал выдаёт одноразовый пропуск на минуту и отправляет его формой (POST, чтобы пропуск не
попадал в адресную строку и журналы) в iframe с ассистентом. Пропуск — `данные.подпись`:
данные — base64url от JSON {"sub", "last_name", "first_name", "middle_name", "role", "exp",
"nonce"}, подпись — HMAC-SHA256 общим секретом PORTAL_SSO_SECRET.

Пользователи портала живут в своём пространстве имён `<sub>@portal.local`: пропуском нельзя
войти в учётную запись, заведённую в ассистенте напрямую, а такой адрес нельзя
зарегистрировать (см. routers/auth.py). Роль берётся из портала при каждом входе — портал
источник правды для своих пользователей.

Без PORTAL_SSO_SECRET вход из портала выключен. PORTAL_ORIGINS — адреса портала через
пробел: только им разрешено встраивать ассистент в iframe (заголовок frame-ancestors).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from threading import Lock

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.auth import create_access_token, get_db, get_password_hash
from app.db import ROLE_ADMIN, ROLE_MANAGER, ROLE_USER, Role, User


PORTAL_DOMAIN = "portal.local"
MIN_SECRET_LEN = 32
MAX_TTL_SECONDS = 120  # пропуск дольше двух минут не принимаем, даже если портал так подписал
_SUB_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_ROLES = {ROLE_ADMIN, ROLE_MANAGER, ROLE_USER}

router = APIRouter()

_used_nonces: dict[str, float] = {}
_nonce_lock = Lock()


def sso_secret() -> str | None:
    secret = (os.getenv("PORTAL_SSO_SECRET") or "").strip()
    return secret if len(secret) >= MIN_SECRET_LEN else None


def portal_origins() -> list[str]:
    return (os.getenv("PORTAL_ORIGINS") or "").split()


def frame_ancestors() -> str:
    return " ".join(["'self'", *portal_origins()])


def portal_email(sub: str) -> str:
    return f"{sub}@{PORTAL_DOMAIN}"


def is_portal_email(email: str) -> bool:
    return email.strip().lower().endswith("@" + PORTAL_DOMAIN)


def sign(payload: dict, secret: str) -> str:
    """Подписать пропуск (так же подписывает портал; здесь — для тестов и проверки связки)."""
    body = base64.urlsafe_b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).rstrip(b"=")
    signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"{body.decode('ascii')}.{signature}"


def verify(token: str, secret: str, now: float | None = None) -> dict:
    """Проверить подпись, срок и одноразовость пропуска. Ошибка — ValueError."""
    now = time.time() if now is None else now
    body, _, signature = token.strip().partition(".")
    if not body or not signature:
        raise ValueError("неверный формат")
    expected = hmac.new(secret.encode("utf-8"), body.encode("ascii", "replace"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise ValueError("неверная подпись")
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except ValueError as exc:
        raise ValueError("неверные данные") from exc
    if not isinstance(payload, dict):
        raise ValueError("неверные данные")

    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or exp < now or exp - now > MAX_TTL_SECONDS:
        raise ValueError("срок пропуска истёк")
    sub = payload.get("sub")
    if not isinstance(sub, str) or not _SUB_RE.match(sub):
        raise ValueError("неверный пользователь")
    nonce = payload.get("nonce")
    if not isinstance(nonce, str) or not 16 <= len(nonce) <= 128:
        raise ValueError("нет одноразового номера")

    with _nonce_lock:
        for used, expires in list(_used_nonces.items()):
            if expires < now:
                del _used_nonces[used]
        if nonce in _used_nonces:
            raise ValueError("пропуск уже использован")
        _used_nonces[nonce] = float(exp)
    return payload


def _text(payload: dict, key: str) -> str:
    value = payload.get(key)
    return value.strip()[:150] if isinstance(value, str) else ""


def _sync_user(db: Session, payload: dict) -> User:
    email = portal_email(payload["sub"])
    last_name, first_name, middle_name = (_text(payload, k) for k in ("last_name", "first_name", "middle_name"))
    full_name = " ".join(part for part in (last_name, first_name, middle_name) if part) or payload["sub"]

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        # Пароль случайный и никому не известен: вход у таких пользователей — только через портал.
        user = User(email=email, hashed_password=get_password_hash(secrets.token_urlsafe(32)))
        db.add(user)
    user.full_name = full_name
    user.last_name = last_name or None
    user.first_name = first_name or None
    user.middle_name = middle_name or None

    role_name = payload.get("role") if payload.get("role") in _ROLES else ROLE_USER
    role = db.query(Role).filter(Role.name == role_name).first()
    user.roles = [role] if role else []
    db.commit()
    db.refresh(user)
    return user


_LANDING = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Вход</title></head>
<body><script>
localStorage.setItem("ai_assistant_token", {token});
location.replace("/");
</script></body></html>"""


@router.post("/sso/portal", response_class=HTMLResponse)
def portal_login(token: str = Form(..., max_length=4096), db: Session = Depends(get_db)) -> HTMLResponse:
    secret = sso_secret()
    if secret is None:
        raise HTTPException(status_code=404, detail="Вход из портала не настроен")
    try:
        payload = verify(token, secret)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=f"Пропуск портала не принят: {exc}") from exc

    user = _sync_user(db, payload)
    access_token = create_access_token({"sub": user.email})
    # JWT — base64url и точки, json.dumps даёт безопасный JS-литерал.
    html = _LANDING.replace("{token}", json.dumps(access_token))
    return HTMLResponse(html, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
