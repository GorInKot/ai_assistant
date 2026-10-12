"""Ответы ассистента на вопросы о себе: распознавание и интеграция в /api/ask."""

from __future__ import annotations

import pytest

from app.services import smalltalk as st
from tests.conftest import register


@pytest.mark.parametrize(
    "text",
    [
        "Что ты умеешь?",
        "что ты умеешь делать",
        "Что вы умеете?",
        "Что ты можешь?",
        "А что ты еще умеешь?",
        "чем ты можешь мне помочь",
        "Чем можешь помочь?",
        "Какие у тебя возможности?",
        "Твои возможности",
        "на что ты способен",
        "Кто ты?",
        "Расскажи о себе",
        "Помощь",
        "Как с тобой работать?",
        "Привет! Что ты умеешь?",
        "что ты умеешь, пожалуйста",
        "ЧТО ТЫ УМЕЕШЬ???",
    ],
)
def test_capabilities_detected(text):
    assert st.detect_meta_kind(text) == st.KIND_CAPABILITIES


@pytest.mark.parametrize("text", ["Привет", "здравствуйте!", "Добрый день", "привет, ассистент"])
def test_greeting_detected(text):
    assert st.detect_meta_kind(text) == st.KIND_GREETING


@pytest.mark.parametrize("text", ["Спасибо", "спасибо большое!", "Благодарю"])
def test_thanks_detected(text):
    assert st.detect_meta_kind(text) == st.KIND_THANKS


@pytest.mark.parametrize(
    "text",
    [
        "Что вы можете сказать про ЕКТП?",  # настоящий вопрос — в поиск
        "Что умеет ЕКТП?",
        "Что ты знаешь про строительный контроль в ЦУС?",
        "Кто отвечает за строительный контроль?",
        "Как использовать шаблон заявки в ЕКТП",
        "Какие функции у системы ЦУС?",
        "Спасибо, а как оформить допуск к работам?",
        "Привет, какие статусы у заявок в ЕКТП?",
        "",
        "   ",
    ],
)
def test_real_questions_not_hijacked(text):
    assert st.detect_meta_kind(text) is None


def test_topics_from_processes():
    topics = st.topics_from_processes(
        ["ЕКТП_Транспорт"] * 3 + ["Лаборатория_ИИ"] + ["Общее", "", "ЦУС_Строительный_контроль"] * 1
    )
    assert topics[0] == "ЕКТП Транспорт"
    assert "Общее" not in topics and "" not in topics
    assert set(topics) == {"ЕКТП Транспорт", "Лаборатория ИИ", "ЦУС Строительный контроль"}


def test_local_llm_detection():
    assert st.is_local_llm("http://127.0.0.1:11434/v1")
    assert st.is_local_llm("http://localhost:1234/v1")
    assert not st.is_local_llm("https://api.groq.com/openai/v1")
    assert not st.is_local_llm(None)


def test_reply_content_matches_real_features():
    text = st.build_reply(st.KIND_CAPABILITIES, documents=37, topics=["ЕКТП Транспорт", "ЦУС Строительный контроль"], local=True)
    for must in ("37 документов", "ЕКТП Транспорт", "Excel", "Word", "PDF", "реестр", "Портал", "локально"):
        assert must in text
    # не обещаем удалённый функционал
    assert "оформлю заявк" not in text.lower()
    assert "локально" not in st.build_reply(st.KIND_CAPABILITIES, local=False)


def test_ask_answers_capabilities_without_llm(client):
    """В тестах LLM не настроен: попади вопрос в RAG — получили бы 503/fallback."""
    headers = register(client, "me@test.local")
    resp = client.post("/api/ask", headers=headers, json={"question": "Что ты умеешь?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "Я корпоративный ассистент" in body["answer"]
    assert body["sources"] == [] and body["no_exact_match"] is False


def test_ask_greeting_and_thanks(client):
    headers = register(client, "me@test.local")
    hello = client.post("/api/ask", headers=headers, json={"question": "Привет!"}).json()
    assert "Здравствуйте" in hello["answer"]
    thanks = client.post("/api/ask", headers=headers, json={"question": "Спасибо"}).json()
    assert "Пожалуйста" in thanks["answer"]
