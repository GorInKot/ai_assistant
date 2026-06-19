"""Тесты обработки файлов, приложенных к сообщению в чате (Этап 6)."""

from __future__ import annotations

import base64
import io

import pytest
from docx import Document
from openpyxl import Workbook, load_workbook

import app.state as state
from app.services import chat_documents as cd
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, auth_headers


# ------------------------------ хелперы файлов ---------------------------


def _xlsx_bytes(headers: list[str], rows: list[list], *, sheet_name: str = "Sheet1") -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(headers)
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _docx_bytes(paragraphs: list[tuple[str, str]]) -> bytes:
    doc = Document()
    for text, style in paragraphs:
        doc.add_paragraph(text, style=style) if style else doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _xlsx_upload(name: str, content: bytes):
    return (name, content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _docx_upload(name: str, content: bytes):
    return (name, content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


class _StubLLM:
    def summarize_text(self, text, **kw):
        return "КРАТКО: " + text[:20]


@pytest.fixture
def headers(client):
    return auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)


# =============== unit-тесты маршрутизации (handle_chat_files) =============


def test_route_single_xlsx_extracts():
    files = [("data.xlsx", _xlsx_bytes(["ID", "Имя"], [[1, "Иван"]]))]
    res = cd.handle_chat_files("", files, _StubLLM())
    assert "data.xlsx" in res.answer
    assert "Иван" in res.answer
    assert res.attachment is None


def test_route_single_xlsx_with_title_row():
    # Лист с объединённой «шапкой» в первой строке + настоящие заголовки ниже.
    wb = Workbook()
    ws = wb.active
    ws.title = "Лист 1 - Сотрудники"
    ws.append(["Сотрудники", None, None])  # титул (как объединённая ячейка)
    ws.append(["№", "ФИО", "Должность"])  # реальные заголовки
    ws.append([1, "Иванов И.И.", "Инженер"])
    ws.append([2, "Петров П.П.", "Техник"])
    buf = io.BytesIO()
    wb.save(buf)

    res = cd.handle_chat_files("покажи содержимое", [("emp.xlsx", buf.getvalue())], _StubLLM())
    # Должно увидеть 3 колонки, а не 1, и ФИО/должности попасть в таблицу.
    assert "3 колонок" in res.answer
    assert "ФИО" in res.answer and "Должность" in res.answer
    assert "Иванов И.И." in res.answer and "Инженер" in res.answer


def test_route_two_xlsx_default_diff():
    a = _xlsx_bytes(["ID", "Имя"], [[1, "Иван"], [2, "Пётр"]])
    b = _xlsx_bytes(["ID", "Имя"], [[1, "Иван"], [2, "Павел"]])
    res = cd.handle_chat_files("", [("a.xlsx", a), ("b.xlsx", b)], _StubLLM())
    assert "Сравнение" in res.answer
    assert "Пётр" in res.answer and "Павел" in res.answer


def test_route_merge_keyword_produces_file():
    a = _xlsx_bytes(["ID"], [[1]])
    b = _xlsx_bytes(["ID"], [[2]])
    res = cd.handle_chat_files("объедини эти файлы", [("a.xlsx", a), ("b.xlsx", b)], _StubLLM())
    assert res.attachment is not None
    assert res.attachment.filename == "merged.xlsx"
    # содержимое — валидный xlsx с двумя строками данных
    wb = load_workbook(io.BytesIO(base64.b64decode(res.attachment.content_base64)))
    assert wb.active.max_row == 3  # header + 2 rows


def test_route_merge_mismatch_no_file():
    a = _xlsx_bytes(["ID", "Имя"], [[1, "Иван"]])
    b = _xlsx_bytes(["ID", "Фамилия"], [[2, "Петров"]])
    res = cd.handle_chat_files("объедини", [("a.xlsx", a), ("b.xlsx", b)], _StubLLM())
    assert res.attachment is None
    assert "структура" in res.answer.lower()


def test_route_single_docx_summary_uses_llm():
    f = [("doc.docx", _docx_bytes([("Большой текст документа.", "")]))]
    res = cd.handle_chat_files("сделай выжимку", f, _StubLLM())
    assert res.answer.startswith("**Выжимка")
    assert "КРАТКО:" in res.answer


def test_route_two_docx_diff():
    a = _docx_bytes([("Первый.", ""), ("Второй.", "")])
    b = _docx_bytes([("Первый.", ""), ("Второй изменён.", "")])
    res = cd.handle_chat_files("сравни", [("a.docx", a), ("b.docx", b)], _StubLLM())
    assert "Сравнение" in res.answer
    assert "✏️" in res.answer


def test_route_mixed_types_rejected():
    with pytest.raises(cd.DocumentError):
        cd.handle_chat_files(
            "",
            [("a.xlsx", _xlsx_bytes(["ID"], [[1]])), ("b.docx", _docx_bytes([("x", "")]))],
            _StubLLM(),
        )


def test_route_unsupported_extension_rejected():
    with pytest.raises(cd.DocumentError):
        cd.handle_chat_files("", [("notes.txt", b"hello")], _StubLLM())


# ===================== интеграция через /api/ask-files ===================


def test_endpoint_requires_auth(client):
    resp = client.post(
        "/api/ask-files",
        data={"message": "что тут"},
        files={"files": _xlsx_upload("x.xlsx", _xlsx_bytes(["a"], [[1]]))},
    )
    assert resp.status_code == 401


def test_endpoint_extract_without_conversation(client, headers):
    resp = client.post(
        "/api/ask-files",
        data={"message": "что в файле"},
        files={"files": _xlsx_upload("data.xlsx", _xlsx_bytes(["ID", "Имя"], [[1, "Иван"]]))},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "Иван" in data["answer"]
    assert data["attachment"] is None


def test_endpoint_summary_persists_in_conversation(client, headers, monkeypatch):
    monkeypatch.setattr(state.llm_service, "summarize_text", lambda text, **kw: "РЕЗЮМЕ ТУТ")
    conv = client.post("/api/conversations", json={}, headers=headers).json()
    resp = client.post(
        "/api/ask-files",
        data={"message": "выжимка", "conversation_id": str(conv["id"])},
        files={"files": _docx_upload("doc.docx", _docx_bytes([("Текст документа.", "")]))},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert "РЕЗЮМЕ ТУТ" in resp.json()["answer"]

    # сообщения (вопрос с пометкой файла + ответ) сохранены в беседе
    detail = client.get(f"/api/conversations/{conv['id']}", headers=headers).json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"]
    assert "📎 doc.docx" in detail["messages"][0]["content"]
