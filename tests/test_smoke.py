"""Smoke-тесты основного пользовательского пути.

Покрывают: health → register/login → защита эндпоинтов → оформление заявки
через чат (slot-filling) → попадание в inbox ответственного → смена статуса.
Также проверяют анонимное обращение (автор скрыт от получателя).

LLM замокан (см. conftest.fake_classify_intent), сеть не используется.
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
    assert client.get("/api/requests/my").status_code == 401
    assert client.get("/api/requests/inbox").status_code == 401
    assert client.post("/api/ask", json={"question": "привет"}).status_code == 401


def test_non_admin_cannot_create_employee(client: TestClient):
    user_headers = register(client, "plain@test.local")
    resp = client.post(
        "/api/admin/employees",
        headers=user_headers,
        json={"email": "x@test.local", "full_name": "X"},
    )
    assert resp.status_code == 403


# ---------- основной сценарий: заявка через чат ----------

def _ask(client: TestClient, headers: dict, text: str) -> dict:
    resp = client.post(
        "/api/ask",
        headers=headers,
        json={"question": text, "session_id": SESSION},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _setup_responsible_manager(client: TestClient, areas: list[str]) -> dict:
    """Регистрирует пользователя-получателя и создаёт связанного с ним Employee.

    Employee создаётся ПОСЛЕ регистрации пользователя — admin-эндпоинт
    автолинкует Employee.user_id по совпадающему email. Только так связанный
    юзер увидит inbox.
    """
    manager_headers = register(
        client, "manager@test.local", last_name="Петров", first_name="Пётр"
    )
    admin_headers = auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    resp = client.post(
        "/api/admin/employees",
        headers=admin_headers,
        json={
            "email": "manager@test.local",
            "full_name": "Петров Пётр",
            "responsibility_area_slugs": areas,
        },
    )
    assert resp.status_code == 201, resp.text
    emp = resp.json()
    # Автолинк сработал: Employee привязан к учётке менеджера.
    assert emp["user_id"] is not None
    return manager_headers


def test_full_request_flow_through_chat(client: TestClient):
    manager_headers = _setup_responsible_manager(client, ["ektp", "legal"])
    requester_headers = register(
        client, "ivan@test.local", last_name="Сидоров", first_name="Иван"
    )

    # 1. Пользователь инициирует заявку на транспорт → ассистент начинает slot-filling.
    r = _ask(client, requester_headers, "нужен транспорт на объект")
    assert "транспорт" in r["answer"].lower()

    # 2. Последовательно заполняем три обязательных слота.
    _ask(client, requester_headers, "2026-06-01 10:00")          # trip_date
    _ask(client, requester_headers, "Строительный объект №5")    # destination
    confirm_prompt = _ask(client, requester_headers, "3")        # passengers_count
    assert "подтвер" in confirm_prompt["answer"].lower()

    # 3. Подтверждаем — заявка создаётся и маршрутизируется ответственному.
    done = _ask(client, requester_headers, "да")
    assert "создана" in done["answer"].lower()

    # 4. Автор видит заявку в /my с раскрытым email-ом инициатора.
    my = client.get("/api/requests/my", headers=requester_headers)
    assert my.status_code == 200
    my_items = my.json()
    assert len(my_items) == 1
    created = my_items[0]
    assert created["type_slug"] == "transport_request"
    assert created["status"] == "new"
    assert created["payload"]["destination"] == "Строительный объект №5"
    assert created["payload"]["passengers_count"] == "3"
    assert created["requester_email"] == "ivan@test.local"

    # 5. Ответственный видит ту же заявку в inbox с раскрытым инициатором.
    inbox = client.get("/api/requests/inbox", headers=manager_headers)
    assert inbox.status_code == 200
    inbox_items = inbox.json()
    assert len(inbox_items) == 1
    assert inbox_items[0]["id"] == created["id"]
    assert inbox_items[0]["requester_name"] == "Сидоров Иван"
    assert inbox_items[0]["is_anonymous"] is False

    # 6. Ответственный меняет статус заявки.
    request_id = created["id"]
    upd = client.put(
        f"/api/requests/{request_id}/status",
        headers=manager_headers,
        json={"status": "in_progress", "comment": "взял в работу"},
    )
    assert upd.status_code == 200
    assert upd.json()["status"] == "in_progress"

    # 7. Изменение видно автору.
    my_after = client.get("/api/requests/my", headers=requester_headers).json()
    assert my_after[0]["status"] == "in_progress"


def test_request_creation_notifies_assigned_employee(client: TestClient, monkeypatch):
    """finalize_request должен звать email-нотификатор с данными получателя."""
    import app.state as state

    calls = []
    monkeypatch.setattr(
        state.email_notifier, "notify_new_request", lambda **kw: calls.append(kw) or True
    )

    manager_headers = _setup_responsible_manager(client, ["ektp"])
    requester_headers = register(client, "ivan@test.local", last_name="Сидоров", first_name="Иван")

    _ask(client, requester_headers, "нужен транспорт")
    _ask(client, requester_headers, "2026-06-01 10:00")
    _ask(client, requester_headers, "Объект")
    _ask(client, requester_headers, "2")
    _ask(client, requester_headers, "да")

    assert len(calls) == 1
    kw = calls[0]
    assert kw["to_email"] == "manager@test.local"
    assert kw["is_anonymous"] is False
    assert kw["requester_name"] == "Сидоров Иван"


def test_requester_cannot_change_status(client: TestClient):
    _setup_responsible_manager(client, ["ektp"])
    requester_headers = register(client, "ivan@test.local", first_name="Иван")

    _ask(client, requester_headers, "нужен транспорт")
    _ask(client, requester_headers, "2026-06-01")
    _ask(client, requester_headers, "Объект")
    _ask(client, requester_headers, "2")
    _ask(client, requester_headers, "да")

    request_id = client.get("/api/requests/my", headers=requester_headers).json()[0]["id"]
    # Автор не является получателем и не admin → менять статус нельзя.
    resp = client.put(
        f"/api/requests/{request_id}/status",
        headers=requester_headers,
        json={"status": "done"},
    )
    assert resp.status_code == 403


def test_anonymous_appeal_hides_requester_in_inbox(client: TestClient):
    manager_headers = _setup_responsible_manager(client, ["legal"])
    requester_headers = register(client, "whistle@test.local", first_name="Аноним")

    _ask(client, requester_headers, "хочу подать анонимное обращение")
    _ask(client, requester_headers, "Нарушение регламента на участке")  # appeal_subject (required)
    _ask(client, requester_headers, "пропустить")                        # appeal_details (optional)
    confirm_prompt = _ask(client, requester_headers, "да")
    assert "создана" in confirm_prompt["answer"].lower()

    inbox = client.get("/api/requests/inbox", headers=manager_headers).json()
    assert len(inbox) == 1
    item = inbox[0]
    assert item["is_anonymous"] is True
    # Ключевое: получатель НЕ видит, кто отправил анонимку.
    assert item["requester_name"] is None
    assert item["requester_email"] is None


def test_cancel_aborts_request_creation(client: TestClient):
    requester_headers = register(client, "ivan@test.local", first_name="Иван")

    started = _ask(client, requester_headers, "нужен транспорт")
    assert "транспорт" in started["answer"].lower()
    cancelled = _ask(client, requester_headers, "отмена")
    assert "прерв" in cancelled["answer"].lower() or "отмен" in cancelled["answer"].lower()

    # Ничего не создалось.
    my = client.get("/api/requests/my", headers=requester_headers).json()
    assert my == []


# ---------- опциональные слоты ----------

def test_optional_slots_can_be_skipped(client: TestClient):
    """training_request: course_name (required) + preferred_dates/comment (optional).

    Пользователь заполняет обязательное поле и пропускает оба необязательных —
    заявка создаётся, пропущенные поля не попадают в payload.
    """
    requester_headers = register(client, "ivan@test.local", first_name="Иван")

    # Старт → спрашивает обязательный course_name.
    _ask(client, requester_headers, "хочу записаться на обучение")

    # Заполняем обязательный слот → дальше идёт необязательный с подсказкой о пропуске.
    optional_prompt = _ask(client, requester_headers, "Python для анализа данных")
    assert "пропустить" in optional_prompt["answer"].lower()

    # Пропускаем оба необязательных слота.
    next_optional = _ask(client, requester_headers, "пропустить")
    assert "пропустить" in next_optional["answer"].lower()
    confirm_prompt = _ask(client, requester_headers, "-")
    assert "подтвер" in confirm_prompt["answer"].lower()

    done = _ask(client, requester_headers, "да")
    assert "создана" in done["answer"].lower()

    created = client.get("/api/requests/my", headers=requester_headers).json()[0]
    assert created["type_slug"] == "training_request"
    assert created["payload"]["course_name"] == "Python для анализа данных"
    # Пропущенные необязательные слоты не сохраняются.
    assert "preferred_dates" not in created["payload"]
    assert "comment" not in created["payload"]


def test_required_slot_cannot_be_skipped(client: TestClient):
    """Попытка пропустить обязательный слот → переспрос, слот не продвигается."""
    requester_headers = register(client, "ivan@test.local", first_name="Иван")

    _ask(client, requester_headers, "нужен транспорт")  # спросит trip_date (required)
    retry = _ask(client, requester_headers, "пропустить")
    assert "обязательно" in retry["answer"].lower()

    # После реального ответа поток продолжается (спрашивает следующий слот).
    nxt = _ask(client, requester_headers, "2026-06-01 10:00")
    assert "обязательно" not in nxt["answer"].lower()
