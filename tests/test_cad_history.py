"""История запросов конструктора деталей: запись, изоляция по пользователям, пересборка без модели."""

from __future__ import annotations

import json

import pytest

import app.state as state
from app.db import SessionLocal
from app.services import cad, cad_history
from tests.conftest import register
from tests.test_cad import PIPES, PLATE, ScriptedLLM, needs_cli

DESCRIPTION = "Стальная пластина 100 на 60 мм, толщина 5, четыре отверстия Ø6"


def _generate(client, headers, llm_reply, description=DESCRIPTION):
    return client.post("/api/cad/generate", headers=headers, json={"description": description})


@pytest.fixture
def llm(monkeypatch):
    """Подменяет ответы модели; возвращает функцию «следующий ответ»."""
    replies: list[str] = []
    monkeypatch.setattr(state.llm_service, "complete_text",
                        lambda messages, json_mode=False: replies.pop(0))
    return replies.append


def _history(client, headers):
    resp = client.get("/api/cad/history", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["items"]


@needs_cli
def test_every_outcome_is_written_to_history(client, llm):
    headers = register(client, "h1@test.local")
    llm(json.dumps(PLATE))
    assert _generate(client, headers, None).status_code == 200
    llm(json.dumps({"clarify": "Укажите толщину."}))
    _generate(client, headers, None, "Пластина 100 на 60")
    llm(json.dumps({"unsupported": "тройники и ответвления труб"}))
    _generate(client, headers, None, "Сделай тройник")
    llm("мусор"), llm("мусор"), llm("мусор")
    assert _generate(client, headers, None, "Пластина").status_code == 422

    items = _history(client, headers)
    assert [i["outcome"] for i in items] == ["error", "unsupported", "clarify", "built"]  # новые первыми
    assert items[3]["description"] == DESCRIPTION and items[3]["title"] == "Деталь"  # название по умолчанию
    assert items[3]["can_rebuild"] is True and items[2]["can_rebuild"] is False
    assert items[1]["detail"] == "тройники и ответвления труб"
    assert "Не удалось составить" in items[0]["detail"]
    assert items[0]["created_at"].endswith("+00:00")  # время с поясом: интерфейс покажет местное


def test_unavailable_model_is_not_recorded(client):
    headers = register(client, "h2@test.local")
    resp = client.post("/api/cad/generate", headers=headers, json={"description": DESCRIPTION})
    assert resp.status_code == 503  # в тестах LLM не настроен
    assert _history(client, headers) == []


@needs_cli
def test_history_is_private_to_each_user(client, llm):
    alice = register(client, "alice@test.local")
    bob = register(client, "bob@test.local")
    llm(json.dumps(PLATE))
    _generate(client, alice, None)
    assert len(_history(client, alice)) == 1 and _history(client, bob) == []
    alice_item = _history(client, alice)[0]
    assert client.delete(f"/api/cad/history/{alice_item['id']}", headers=bob).status_code == 404
    assert client.post(f"/api/cad/history/{alice_item['id']}/rebuild", headers=bob).status_code == 404
    assert len(_history(client, alice)) == 1


@needs_cli
def test_delete_removes_own_record(client, llm):
    headers = register(client, "h3@test.local")
    llm(json.dumps(PLATE))
    _generate(client, headers, None)
    item = _history(client, headers)[0]
    assert client.delete(f"/api/cad/history/{item['id']}", headers=headers).status_code == 200
    assert _history(client, headers) == []
    assert client.delete(f"/api/cad/history/{item['id']}", headers=headers).status_code == 404


@needs_cli
def test_rebuild_uses_stored_spec_without_the_model(client, llm, monkeypatch):
    headers = register(client, "h4@test.local")
    llm(json.dumps(PIPES))
    first = _generate(client, headers, None, "Две трубы под углом 60°").json()
    item = _history(client, headers)[0]

    def model_must_not_be_called(*args, **kwargs):
        raise AssertionError("пересборка не должна обращаться к модели")

    monkeypatch.setattr(state.llm_service, "complete_text", model_must_not_be_called)
    resp = client.post(f"/api/cad/history/{item['id']}/rebuild", headers=headers)
    assert resp.status_code == 200, resp.text
    again = resp.json()
    assert again["metrics"] == first["metrics"] and again["summary"] == first["summary"]
    assert again["assumptions"] == first["assumptions"]  # пометки сохраняются
    assert all(c["ok"] for c in again["checks"])


@needs_cli
def test_rebuild_is_refused_for_requests_without_a_part(client, llm):
    headers = register(client, "h5@test.local")
    llm(json.dumps({"unsupported": "шестерни"}))
    _generate(client, headers, None, "Шестерня")
    item = _history(client, headers)[0]
    assert item["can_rebuild"] is False
    resp = client.post(f"/api/cad/history/{item['id']}/rebuild", headers=headers)
    assert resp.status_code == 400


def test_old_records_are_pruned(client, monkeypatch):
    headers = register(client, "h6@test.local")
    monkeypatch.setattr(cad_history, "MAX_PER_USER", 3)
    with SessionLocal() as db:
        from app.db import User
        user_id = db.query(User).filter(User.email == "h6@test.local").one().id
        for i in range(5):
            cad_history.record(db, user_id, f"запрос {i}", cad_history.OUTCOME_CLARIFY, detail="?")
    items = client.get("/api/cad/history?limit=3", headers=headers).json()["items"]
    assert [i["description"] for i in items] == ["запрос 4", "запрос 3", "запрос 2"]


def test_history_limit_is_validated_and_auth_required(client):
    headers = register(client, "h7@test.local")
    assert client.get("/api/cad/history?limit=0", headers=headers).status_code == 422
    assert client.get("/api/cad/history?limit=100000", headers=headers).status_code == 422
    assert client.get("/api/cad/history").status_code == 401
    assert client.post("/api/cad/history/1/rebuild").status_code == 401


def test_corrupt_stored_spec_gives_a_clear_error(client):
    headers = register(client, "h8@test.local")
    with SessionLocal() as db:
        from app.db import User
        user_id = db.query(User).filter(User.email == "h8@test.local").one().id
        cad_history.record(db, user_id, "битая запись", cad_history.OUTCOME_BUILT,
                           title="x", spec={"shape": "box", "length": -5})
    item = _history(client, headers)[0]
    if cad.is_available(state.settings.solidcraft_dir):
        resp = client.post(f"/api/cad/history/{item['id']}/rebuild", headers=headers)
        assert resp.status_code == 422 and "больше не проходит проверку" in resp.json()["detail"]
