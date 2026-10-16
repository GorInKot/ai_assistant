"""Вход из портала: одноразовый подписанный пропуск, отдельное пространство имён, роль из портала."""

from __future__ import annotations

import re
import secrets
import time

import pytest

from app.routers import portal_sso
from tests.conftest import register

SECRET = "s" * 40


@pytest.fixture(autouse=True)
def _sso_env(monkeypatch):
    monkeypatch.setenv("PORTAL_SSO_SECRET", SECRET)
    monkeypatch.setenv("PORTAL_ORIGINS", "http://127.0.0.1:8080")


def _pass(sub="ivanov", role="user", ttl=60, secret=SECRET, **extra) -> str:
    payload = {"sub": sub, "last_name": "Иванов", "first_name": "Иван", "middle_name": "Иванович",
               "role": role, "exp": int(time.time()) + ttl, "nonce": secrets.token_hex(16), **extra}
    return portal_sso.sign(payload, secret)


def _login(client, token: str):
    return client.post("/sso/portal", data={"token": token})


def _jwt(resp) -> str:
    return re.search(r'setItem\("ai_assistant_token", "([^"]+)"\)', resp.text).group(1)


def test_valid_pass_creates_portal_user_and_logs_in(client):
    resp = _login(client, _pass())
    assert resp.status_code == 200, resp.text
    assert resp.headers["cache-control"] == "no-store"
    me = client.get("/api/user/profile", headers={"Authorization": f"Bearer {_jwt(resp)}"}).json()
    assert me["email"] == "ivanov@portal.local" and me["full_name"] == "Иванов Иван Иванович"
    assert me["roles"] == ["user"]


def test_forged_signature_is_rejected(client):
    assert _login(client, _pass(secret="x" * 40)).status_code == 403


def test_expired_and_too_long_passes_are_rejected(client):
    assert _login(client, _pass(ttl=-5)).status_code == 403
    assert _login(client, _pass(ttl=portal_sso.MAX_TTL_SECONDS + 60)).status_code == 403


def test_pass_is_single_use(client):
    token = _pass()
    assert _login(client, token).status_code == 200
    assert _login(client, token).status_code == 403


@pytest.mark.parametrize("sub", ["", "a@b", "../admin", "Иванов", "x" * 65])
def test_bad_subject_is_rejected(client, sub):
    assert _login(client, _pass(sub=sub)).status_code == 403


def test_disabled_without_secret(client, monkeypatch):
    monkeypatch.setenv("PORTAL_SSO_SECRET", "short")
    assert _login(client, _pass()).status_code == 404


def test_role_follows_portal_on_each_login(client):
    first = _login(client, _pass(sub="petrov", role="manager"))
    headers = {"Authorization": f"Bearer {_jwt(first)}"}
    assert client.get("/api/user/profile", headers=headers).json()["roles"] == ["manager"]
    _login(client, _pass(sub="petrov", role="user"))
    assert client.get("/api/user/profile", headers=headers).json()["roles"] == ["user"]
    _login(client, _pass(sub="petrov", role="superuser"))  # неизвестная роль — обычный пользователь
    assert client.get("/api/user/profile", headers=headers).json()["roles"] == ["user"]


def test_portal_namespace_cannot_be_registered_directly(client):
    resp = client.post("/api/auth/register", json={
        "email": "sidorov@portal.local", "password": "pass-12345", "confirm_password": "pass-12345",
        "last_name": "Сидоров", "first_name": "Сидор", "division": "ЦА",
    })
    assert resp.status_code == 400


def test_pass_cannot_reach_locally_registered_account(client):
    register(client, "local-user@test.local")
    resp = _login(client, _pass(sub="local-user"))
    me = client.get("/api/user/profile", headers={"Authorization": f"Bearer {_jwt(resp)}"}).json()
    assert me["email"] == "local-user@portal.local"  # отдельная учётная запись, не local-user@test.local


def test_pass_made_by_portal_bridge_is_accepted(client):
    # пропуск, который выдаёт надстройка портала (без Django), ассистент понимает
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integrations" / "portal"))
    try:
        from portal_bridge.sso_pass import make_pass, subject
    finally:
        sys.path.pop(0)

    assert subject("ivanov.ii", 7) == "ivanov.ii" and subject("Иванов", 7) == "user-7"
    token = make_pass(sub=subject("Иванов", 7), last_name="Иванов", first_name="Иван", middle_name="",
                      role="admin", secret=SECRET)
    me = client.get("/api/user/profile", headers={"Authorization": f"Bearer {_jwt(_login(client, token))}"}).json()
    assert me["email"] == "user-7@portal.local" and me["roles"] == ["admin"]


def test_only_portal_may_frame_the_assistant(client):
    policy = client.get("/api/health").headers["content-security-policy"]
    assert policy == "frame-ancestors 'self' http://127.0.0.1:8080"


def _next(resp) -> str:
    return re.search(r"location\.replace\((\"[^\"]*\")\)", resp.text).group(1)


def test_after_login_goes_to_requested_section(client):
    assert _next(_login(client, _pass())) == '"/"'
    resp = client.post("/sso/portal", data={"token": _pass(), "next": "/tools"})
    assert resp.status_code == 200 and _next(resp) == '"/tools"'


@pytest.mark.parametrize("target", ["https://evil.example", "//evil.example", "/tools?x=1", "/api/health",
                                    "javascript:alert(1)", "/tools</script>"])
def test_unknown_next_falls_back_to_chat(client, target):
    resp = client.post("/sso/portal", data={"token": _pass(), "next": target})
    assert resp.status_code == 200 and _next(resp) == '"/"'


def test_tools_section_is_served_like_chat(client):
    root, tools = client.get("/"), client.get("/tools")
    assert tools.status_code == root.status_code
    if root.status_code == 200:
        assert tools.content == root.content
