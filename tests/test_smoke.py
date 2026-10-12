"""Smoke-тесты основного пользовательского пути.

Покрывают: health → register/login → защита эндпоинтов → вопрос в чат.
Оформление заявок из ассистента убрано (оно живёт в Портале), поэтому здесь же
проверяется, что соответствующих эндпоинтов нет, а просьба «оформи заявку»
обрабатывается как обычный вопрос.

LLM по сети не ходит: KB в тестах пустой, до генерации ответа дело не доходит.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, auth_headers, register


SESSION = "smoke-session"


# ---------- инфраструктура / auth ----------

def test_health(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "documents" in body and "chunks" in body


def test_admin_seeded_and_can_login(client: TestClient):
    headers = auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    resp = client.get("/api/user/profile", headers=headers)
    assert resp.status_code == 200
    assert "admin" in resp.json()["roles"]


def test_register_then_login(client: TestClient):
    register(client, "newbie@test.local")
    headers = auth_headers(client, "newbie@test.local", "user-pass-123")
    resp = client.get("/api/user/profile", headers=headers)
    assert resp.status_code == 200
    profile = resp.json()
    assert profile["email"] == "newbie@test.local"
    # Не первый пользователь (admin засижен из env) → обычная роль user, не admin.
    assert profile["roles"] == ["user"]


def test_login_wrong_password_rejected(client: TestClient):
    register(client, "u1@test.local")
    resp = client.post(
        "/api/auth/login", json={"email": "u1@test.local", "password": "wrong"}
    )
    assert resp.status_code == 401


def test_protected_endpoints_require_auth(client: TestClient):
    assert client.post("/api/ask", json={"question": "привет"}).status_code == 401
    assert client.get("/api/conversations").status_code == 401


# ---------- заявки вынесены в Портал ----------

def test_request_and_employee_endpoints_are_gone(client: TestClient):
    headers = auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    for path in (
        "/api/requests/my",
        "/api/requests/inbox",
        "/api/admin/employees",
        "/api/admin/request-types",
        "/api/admin/responsibility-areas",
    ):
        assert client.get(path, headers=headers).status_code == 404, path


def test_request_phrase_is_answered_as_plain_question(client: TestClient):
    """«Оформи заявку» больше не запускает slot-filling: это обычный вопрос к KB."""
    headers = register(client, "ivan@test.local")
    resp = client.post(
        "/api/ask",
        headers=headers,
        json={"question": "Оформи заявку на транспорт на завтра", "session_id": SESSION},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # KB в тестах пуст → обычный fallback, а не вопрос «Куда поедете?» / «создана заявка».
    assert "заявка" not in body["answer"].lower() or "нет точной информации" in body["answer"].lower()
    assert body["attachment"] is None


def test_assistant_has_no_request_tables(client: TestClient):
    from sqlalchemy import inspect

    from app.db import engine

    tables = set(inspect(engine).get_table_names())
    assert {"users", "roles", "conversations", "messages"} <= tables
    assert not tables & {
        "requests", "request_events", "request_types", "request_type_slots",
        "employees", "responsibility_areas", "responsibilities",
    }
