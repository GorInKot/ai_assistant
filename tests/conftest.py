"""Общие фикстуры для smoke-тестов.

Изоляция: тесты гоняются на отдельной временной SQLite-базе (env DATABASE_URL
выставляется ДО первого импорта app.*). LLM не вызывается по сети — метод
classify_intent подменяется детерминированным фейком (см. fake_classify_intent).

Зачем фейк, а не реальный LLM: smoke-тесты должны быть быстрыми, офлайн и
воспроизводимыми. Сетевой intent-classifier сделал бы их флаки и платными.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# --- env ДОЛЖЕН быть выставлен до импорта app.db / app.state / app.main ---
_TMP_DB_FD, _TMP_DB_PATH = tempfile.mkstemp(prefix="smoke_test_", suffix=".db")
os.close(_TMP_DB_FD)
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB_PATH}"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["INITIAL_ADMIN_EMAIL"] = "admin@test.local"
os.environ["INITIAL_ADMIN_PASSWORD"] = "admin-pass-123"
# Без YC-ключей KB работает на BM25, без сети (эмбеддер отключён).
os.environ.pop("YC_API_KEY", None)
os.environ.pop("YC_FOLDER_ID", None)

from fastapi.testclient import TestClient  # noqa: E402

import app.state as state  # noqa: E402
from app.db import Base, engine, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.request_catalog import reload_catalog  # noqa: E402


ADMIN_EMAIL = os.environ["INITIAL_ADMIN_EMAIL"]
ADMIN_PASSWORD = os.environ["INITIAL_ADMIN_PASSWORD"]


def fake_classify_intent(question: str, catalog_summary) -> dict:
    """Детерминированная замена LLM-классификатора интентов.

    Маппинг подобран под сценарии smoke-тестов. Ответы на слоты (даты, адреса,
    числа) попадают в ветку 'qa' — этого достаточно: в slot-filling важно лишь,
    что интент НЕ confirm/cancel.
    """
    q = question.strip().lower()
    if q in {"да", "yes", "подтверждаю", "ага"}:
        return {"intent": "confirm_yes"}
    if q in {"нет", "no", "отмена", "отменить"}:
        return {"intent": "cancel"}
    if "транспорт" in q or "машин" in q:
        return {"intent": "create_request", "request_type": "transport_request"}
    if "обучение" in q or "курс" in q:
        return {"intent": "create_request", "request_type": "training_request"}
    if "анонимн" in q:
        return {"intent": "create_request", "request_type": "anonymous_appeal"}
    return {"intent": "qa"}


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    """LLM никогда не ходит по сети в тестах: подменяем classify_intent.

    generate_answer не мокаем — в QA-тесте KB пустой, поэтому до генерации
    дело не доходит (срабатывает fallback 'нет точной информации').
    """
    monkeypatch.setattr(state.llm_service, "classify_intent", fake_classify_intent)


@pytest.fixture(autouse=True)
def clean_db():
    """Чистое состояние БД перед каждым тестом: drop → create → reseed.

    Также сбрасываем in-memory dialog_state (иначе pending-заявки от прошлого
    теста протекают через переиспользуемые user_id) и кэш каталога заявок.
    """
    Base.metadata.drop_all(bind=engine)
    init_db()  # пересоздаёт схему + сидит роли/области/типы заявок + INITIAL_ADMIN
    reload_catalog()
    state.dialog_state._clarifications.clear()
    state.dialog_state._requests.clear()
    yield


@pytest.fixture
def client():
    # Без context-manager'а: startup-событие (построение KB) не запускается —
    # для смоук-сценариев заявок KB не нужен, экономим секунды на индексации.
    return TestClient(app)


# ---------- helpers ----------

def auth_headers(client: TestClient, email: str, password: str) -> dict[str, str]:
    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def register(
    client: TestClient,
    email: str,
    password: str = "user-pass-123",
    *,
    last_name: str = "Иванов",
    first_name: str = "Иван",
    division: str = "ЦА",
    subdivision: str | None = None,
) -> dict[str, str]:
    resp = client.post(
        "/api/auth/register",
        json={
            "email": email,
            "password": password,
            "confirm_password": password,
            "last_name": last_name,
            "first_name": first_name,
            "division": division,
            "subdivision": subdivision,
        },
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}
