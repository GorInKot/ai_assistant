"""Сквозная проверка: ответ модели → цитаты и неподтверждённые утверждения в API и в истории."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.state as state
from app.kb import KnowledgeBaseIndex
from app.services import ask_service
from tests.conftest import register

KB_TEXT = """# Статусы заявки в ЕКТП

Заявка на транспорт в ЕКТП проходит несколько статусов.

Черновик — статус заявки после создания, до размещения на обработку.
На обработке — заявка размещена и ожидает проверки диспетчером.
Принята — заявка согласована ответственным лицом.
"""

GOOD_AND_BAD_ANSWER = (
    "Заявка в ЕКТП после создания получает статус Черновик.\n"
    "После размещения заявка находится на обработке и ожидает проверки диспетчером.\n"
    "Срок рассмотрения заявки составляет 14 рабочих дней."
)


@pytest.fixture
def kb_and_llm(tmp_path: Path, monkeypatch):
    folder = tmp_path / "ЕКТП_Транспорт"
    folder.mkdir()
    (folder / "Статусы.md").write_text(KB_TEXT, encoding="utf-8")
    kb = KnowledgeBaseIndex(tmp_path)
    kb.build()
    monkeypatch.setattr(ask_service, "kb_index", kb)
    monkeypatch.setattr(state.llm_service, "rerank_results", lambda q, cands, top_n=16: list(cands)[:top_n])
    monkeypatch.setattr(state.llm_service, "generate_answer", lambda q, ctx, intent="procedure": GOOD_AND_BAD_ANSWER)
    return kb


def _ask(client, headers, **extra):
    resp = client.post(
        "/api/ask",
        headers=headers,
        json={"question": "Какие статусы бывают у заявки в ЕКТП?", **extra},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_answer_carries_citations_and_flags_invented_claim(client, kb_and_llm):
    headers = register(client, "g@test.local")
    body = _ask(client, headers)
    assert body["no_exact_match"] is False, body["answer"]
    g = body["grounding"]
    assert g["total"] == 3 and g["supported"] >= 2
    assert len(g["unsupported"]) == 1 and "14" in g["unsupported"][0]
    cite = g["citations"][0]
    assert cite["file_name"] == "Статусы.md"
    assert cite["quote"] in KB_TEXT  # цитата дословная


def test_grounding_is_persisted_in_conversation_history(client, kb_and_llm):
    headers = register(client, "g2@test.local")
    conv = client.post("/api/conversations", headers=headers, json={}).json()
    _ask(client, headers, conversation_id=conv["id"])
    detail = client.get(f"/api/conversations/{conv['id']}", headers=headers).json()
    assistant = [m for m in detail["messages"] if m["role"] == "assistant"][0]
    assert assistant["grounding"]["total"] == 3
    assert any("14" in u for u in assistant["grounding"]["unsupported"])
    user = [m for m in detail["messages"] if m["role"] == "user"][0]
    assert user["grounding"] is None


def test_fallback_answers_have_no_grounding(client):
    headers = register(client, "g3@test.local")
    resp = client.post("/api/ask", headers=headers, json={"question": "Какая погода на Марсе?"})
    assert resp.status_code == 200
    assert resp.json()["grounding"] is None


def test_smalltalk_has_no_grounding(client):
    headers = register(client, "g4@test.local")
    resp = client.post("/api/ask", headers=headers, json={"question": "Что ты умеешь?"})
    assert resp.json()["grounding"] is None
