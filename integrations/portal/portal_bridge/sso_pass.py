"""Одноразовый пропуск в ассистент. Только стандартная библиотека — модуль проверяется
тестами ассистента без Django.

Формат — `данные.подпись`: данные — base64url от JSON {"sub", "last_name", "first_name",
"middle_name", "role", "exp", "nonce"}, подпись — HMAC-SHA256 общим секретом. Ассистент
проверяет его в app/routers/portal_sso.py.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time

PASS_TTL_SECONDS = 60
_SAFE_SUB = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def subject(username: str, pk: int) -> str:
    """Имя пользователя в ассистенте: логин портала, если он из безопасных символов, иначе номер."""
    return username if _SAFE_SUB.match(username or "") else f"user-{pk}"


def make_pass(*, sub: str, last_name: str, first_name: str, middle_name: str, role: str,
              secret: str, now: float | None = None) -> str:
    now = time.time() if now is None else now
    payload = {
        "sub": sub,
        "last_name": last_name,
        "first_name": first_name,
        "middle_name": middle_name,
        "role": role,
        "exp": int(now) + PASS_TTL_SECONDS,
        "nonce": secrets.token_hex(16),
    }
    body = base64.urlsafe_b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).rstrip(b"=")
    signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"{body.decode('ascii')}.{signature}"
