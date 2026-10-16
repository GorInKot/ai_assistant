"""Ассистент видит страницу портала: очистка и объём содержимого, ответ по странице, журнал и история."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.state as state
from app.db import Message, QaLog, SessionLocal
from app.kb import KnowledgeBaseIndex
from app.services import ask_service, page_context
from app.services.page_context import PageContext
from tests.conftest import register

# Выдуманная страница реестра — как её собирает widget.js.
PAGE = {
    "path": "/metrology/",
    "title": "Портал",
    "heading": "Реестр средств измерений",
    "tab": "Реестр",
    "fields": [
        {"label": "Филиал", "value": "Все филиалы"},
        {"label": "Статус", "value": "Просрочена поверка"},
        {"label": "Пароль", "value": "qwerty-123"},
        {"label": "csrfmiddlewaretoken", "value": "x"},
        {"label": "Комментарий", "value": "ключ " + "A1b2C3d4" * 6},
    ],
    "selected": "",
    "tables": [{
        "caption": "",
        "headers": ["Наименование СИ", "Зав. №", "Срок поверки"],
        "rows": [["Манометр МП-100", "1234", "01.11.2026"], ["Термометр ТЛ-4", "77", "15.12.2026"]],
        "total_rows": 2,
    }],
    "errors": ["Не заполнено поле «Подразделение»"],
    "text": "Реестр средств измерений\nНайдено: 2\nНайти",
}

KB_TEXT = """# Статусы заявки в ЕКТП

Заявка на транспорт в ЕКТП проходит несколько статусов: Черновик, На обработке, Принята.
"""


def _ctx(**over) -> PageContext:
    return PageContext(**{**PAGE, **over})


# ---------- очистка и объём ----------


def test_passwords_service_fields_and_keys_are_never_sent():
    text = page_context.render(_ctx())
    assert "qwerty" not in text and "Пароль" not in text and "csrf" not in text
    assert "A1b2C3d4" not in text
    assert "- Филиал: Все филиалы" in text and "Просрочена поверка" in text


def test_block_is_limited_and_keeps_the_important_part_first():
    huge = _ctx(text="слово " * 990, tables=[{**PAGE["tables"][0], "rows": [["x" * 80] * 12] * 20}])
    text = page_context.render(huge)
    assert len(text) <= page_context.MAX_CHARS
    assert text.startswith("Раздел портала: Метрология")
    assert "Не заполнено поле" in text and "- Филиал: Все филиалы" in text


def test_tables_errors_and_tab_are_rendered():
    text = page_context.render(_ctx())
    assert "Активная вкладка: Реестр" in text
    assert "| Наименование СИ | Зав. № | Срок поверки |" in text and "| Манометр МП-100 | 1234 | 01.11.2026 |" in text
    assert "Сообщения об ошибках на странице:" in text


@pytest.mark.parametrize("path,section", [
    ("/", "Главная портала"), ("/metrology/", "Метрология"), ("/metrology/add/", "Метрология"),
    ("/siz/", "СИЗ"), ("/protocol/", "Протокол"), ("/assistant/tools/", "Конструктор и чертежи"),
    ("/unknown/", "Портал"),
])
def test_section_by_path(path, section):
    assert page_context.section_for(path) == section


def test_label_is_short_and_without_page_text():
    assert page_context.label(_ctx()) == "Метрология → Реестр средств измерений"
    assert page_context.label(_ctx(path="/", heading="", title="")) == "Главная портала"


@pytest.mark.parametrize("question", [
    "что представлено на экране?", "Что на этой странице?", "что здесь можно сделать",
    "что означает это поле?", "почему ошибка на странице", "где я?", "что это за раздел", "что за таблица?",
])
def test_screen_questions(question):
    assert page_context.is_screen_question(question)


@pytest.mark.parametrize("question", [
    "Как подать заявку на транспорт?", "Какие статусы бывают у заявки в ЕКТП?", "Что такое ЦУС?",
])
def test_ordinary_questions_are_not_screen_questions(question):
    assert not page_context.is_screen_question(question)


# ---------- ответ по странице ----------


@pytest.fixture
def kb_and_llm(tmp_path: Path, monkeypatch):
    folder = tmp_path / "ЕКТП_Транспорт"
    folder.mkdir()
    (folder / "Статусы.md").write_text(KB_TEXT, encoding="utf-8")
    kb = KnowledgeBaseIndex(tmp_path)
    kb.build()
    monkeypatch.setattr(ask_service, "kb_index", kb)
    monkeypatch.setattr(state.llm_service, "rerank_results", lambda q, cands, top_n=16: list(cands)[:top_n])
    calls = {}

    def page_answer(question, block, ctx):
        calls["block"] = block
        return "На экране реестр средств измерений: Манометр МП-100 и Термометр ТЛ-4."

    monkeypatch.setattr(state.llm_service, "generate_page_answer", page_answer)
    monkeypatch.setattr(state.llm_service, "generate_answer", lambda q, ctx, intent="procedure": "Черновик, На обработке, Принята.")
    return calls


def _ask(client, headers, question, page=PAGE, conversation_id=None):
    resp = client.post("/api/ask", headers=headers, json={
        "question": question, "page_context": page, "conversation_id": conversation_id,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_screen_question_is_answered_from_the_page(client, kb_and_llm):
    headers = register(client, "p@test.local")
    body = _ask(client, headers, "что представлено на экране?")
    assert body["kind"] == "page" and "Манометр" in body["answer"]
    assert "Манометр МП-100" in kb_and_llm["block"] and "qwerty" not in kb_and_llm["block"]
    assert body["grounding"]["citations"][0]["file_name"] == ask_service.PAGE_SOURCE


def test_without_page_the_same_question_is_not_a_page_answer(client, kb_and_llm):
    headers = register(client, "np@test.local")
    resp = client.post("/api/ask", headers=headers, json={"question": "что представлено на экране?"})
    assert resp.json()["kind"] != "page"


def test_knowledge_questions_still_use_the_knowledge_base(client, kb_and_llm):
    headers = register(client, "kb@test.local")
    body = _ask(client, headers, "Какие статусы бывают у заявки в ЕКТП?")
    assert body["kind"] == "answer" and "block" not in kb_and_llm


def test_unanswerable_question_from_a_page_falls_back_to_the_page(client, kb_and_llm):
    headers = register(client, "fb@test.local")
    body = _ask(client, headers, "Сколько приборов с просроченной поверкой?")
    assert body["kind"] == "page"


def test_log_gets_section_and_history_gets_only_a_label(client, kb_and_llm):
    headers = register(client, "log@test.local")
    conv = client.post("/api/conversations", headers=headers, json={}).json()
    _ask(client, headers, "что на экране?", conversation_id=conv["id"])
    db = SessionLocal()
    try:
        row = db.query(QaLog).order_by(QaLog.id.desc()).first()
        assert row.portal_section == "Метрология" and row.kind == "page"
        user_msg = db.query(Message).filter(Message.conversation_id == conv["id"], Message.role == "user").one()
        assert user_msg.page_label == "Метрология → Реестр средств измерений"
        assert "Манометр" not in (user_msg.content + (user_msg.page_label or ""))
    finally:
        db.close()
    detail = client.get(f"/api/conversations/{conv['id']}", headers=headers).json()
    assert detail["messages"][0]["page_label"] == "Метрология → Реестр средств измерений"


def test_preview_shows_exactly_what_is_sent(client):
    headers = register(client, "pv@test.local")
    resp = client.post("/api/page-context/preview", headers=headers, json=PAGE)
    body = resp.json()
    assert body["label"] == "Метрология → Реестр средств измерений" and body["rows"] == 2
    assert body["text"] == page_context.render(PageContext(**PAGE))
    assert client.post("/api/page-context/preview", json=PAGE).status_code == 401


def test_oversized_page_is_rejected_not_truncated_silently(client):
    headers = register(client, "big@test.local")
    resp = client.post("/api/ask", headers=headers, json={"question": "что на экране?", "page_context": {**PAGE, "text": "x" * 7000}})
    assert resp.status_code == 422


# ---------- введённые данные и расчёты ----------

FORM = {**PAGE, "path": "/metrology/add/", "heading": "Добавить СИ", "tables": [], "errors": [], "fields": [
    {"label": "Наименование оборудования и СИ", "value": "Манометр МП-100"},
    {"label": "Дата текущей поверки/калибровки", "value": "2026-10-01"},
    {"label": "Дата следующей поверки", "value": "01.09.2026"},
]}


@pytest.mark.parametrize("question", [
    "Сколько месяцев между датами поверки, которые я указал?",
    "Правильно ли я заполнил форму?",
    "Подходят ли мои данные?",
    "Проверь введённые значения",
    "Какой срок у даты следующей поверки?",
])
def test_questions_about_entered_data_or_named_fields_go_to_the_page(question):
    assert page_context.is_page_question(question, PageContext(**FORM))


@pytest.mark.parametrize("question", ["Что такое поверка?", "Как подать заявку на транспорт?"])
def test_general_questions_stay_with_the_knowledge_base(question):
    assert not page_context.is_page_question(question, PageContext(**FORM))


def test_dates_are_compared_and_wrong_order_is_flagged():
    from datetime import date

    facts = page_context.facts(PageContext(**FORM), today=date(2026, 10, 16))
    assert facts[0] == "Сегодня: 16.10.2026"
    assert "«Дата текущей поверки/калибровки» = 01.10.2026; относительно сегодняшней даты — 15 дн. назад" in facts
    assert "Промежуток между «Дата текущей поверки/калибровки» и «Дата следующей поверки»: 1 мес." in facts
    assert ("Ошибка порядка дат: «Дата следующей поверки» раньше «Дата текущей поверки/калибровки», "
            "а должна быть позже") in facts


def test_right_order_is_not_flagged():
    from datetime import date

    fields = [{"label": "Дата текущей поверки", "value": "01.10.2026"}, {"label": "Дата следующей поверки", "value": "01.10.2027"}]
    facts = page_context.facts(PageContext(path="/metrology/add/", fields=fields), today=date(2026, 10, 16))
    assert "Промежуток между «Дата текущей поверки» и «Дата следующей поверки»: 12 мес." in facts
    assert "Порядок дат: «Дата текущей поверки» раньше «Дата следующей поверки»" in facts
    assert not any("Ошибка" in f for f in facts)


@pytest.mark.parametrize("a,b,expected", [
    ("2026-01-31", "2026-02-28", "1 мес."), ("2026-01-15", "2026-03-20", "2 мес. 5 дн."), ("2026-05-01", "2026-05-01", "0 дн."),
])
def test_span(a, b, expected):
    assert page_context.span(page_context.parse_date(a), page_context.parse_date(b)) == expected


def test_numeric_columns_are_summed_and_facts_are_in_the_block():
    ctx = PageContext(path="/siz/", tables=[{"caption": "", "headers": ["СИЗ", "Кол-во"],
                                             "rows": [["Каска", "2"], ["Перчатки", "10,5"]], "total_rows": 2}])
    assert any(f.endswith("сумма по колонке «Кол-во» в показанных строках (2) = 12,5") for f in page_context.facts(ctx))
    assert "Посчитано ассистентом по значениям на странице (точно):" in page_context.render(ctx)


def test_entered_data_question_is_answered_from_the_page(client, kb_and_llm):
    headers = register(client, "in@test.local")
    body = _ask(client, headers, "Сколько месяцев между датами поверки, которые я указал?", page=FORM)
    assert body["kind"] == "page" and body["grounding"]["basis"] == "page"
    assert "Промежуток между «Дата текущей поверки/калибровки» и «Дата следующей поверки»: 1 мес." in kb_and_llm["block"]


def test_dates_from_date_inputs_are_shown_in_one_format():
    text = page_context.render(PageContext(**FORM))
    assert "- Дата текущей поверки/калибровки: 01.10.2026" in text and "2026-10-01" not in text
