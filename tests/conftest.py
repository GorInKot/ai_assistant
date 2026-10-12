"""Общие фикстуры для smoke-тестов.

Изоляция: тесты гоняются на отдельной временной SQLite-базе (env DATABASE_URL
выставляется ДО первого импорта app.*). В тестах нет ключей LLM, поэтому по сети
ничего не вызывается: smoke-тесты быстрые, офлайн и воспроизводимые.
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
# Тесты не должны зависеть от личного .env разработчика (локальная модель, ключи):
# пустое значение в окружении приоритетнее файла .env. Без эмбеддера KB работает на
# BM25, без LLM — любые попытки генерации дают 503, сеть и модели не вызываются.
for _name in (
    "YC_API_KEY", "YC_FOLDER_ID",
    "EMBEDDINGS_BASE_URL", "EMBEDDINGS_MODEL", "EMBEDDINGS_API_KEY",
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_REASONING_EFFORT",
    "OPENAI_API_KEY", "OPENAI_MODEL",
):
    os.environ[_name] = ""

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

import app.state as state  # noqa: E402
from app.db import Base, engine, init_db  # noqa: E402
from app.main import app  # noqa: E402


ADMIN_EMAIL = os.environ["INITIAL_ADMIN_EMAIL"]
ADMIN_PASSWORD = os.environ["INITIAL_ADMIN_PASSWORD"]


@pytest.fixture(autouse=True)
def clean_db():
    """Чистое состояние БД перед каждым тестом: drop → create → reseed.

    Также сбрасываем in-memory dialog_state (иначе незаконченные уточнения от
    прошлого теста протекают через переиспользуемые user_id).
    """
    Base.metadata.drop_all(bind=engine)
    # init_db теперь гоняет Alembic: после drop_all таблица alembic_version
    # осталась бы и upgrade head счёл бы схему актуальной (таблиц нет!). Сносим
    # её, чтобы _run_migrations пошёл по ветке «чистая БД» и пересоздал схему.
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    init_db()  # миграции до head + сиды ролей + INITIAL_ADMIN
    state.dialog_state._clarifications.clear()
    yield


@pytest.fixture
def client():
    # Без context-manager'а: startup-событие (построение KB) не запускается —
    # для смоук-сценариев KB не нужен, экономим секунды на индексации.
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
