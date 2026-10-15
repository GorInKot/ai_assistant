"""Что за документ: вид, основание, номер и дата, предложенные действия. Все документы выдуманные."""

from __future__ import annotations

import io
import json

import pytest
from docx import Document

from app.services import chat_documents as cd
from app.services import doc_identify as di
from app.services.document_processing import repair_cp1251
from tests.test_spec_registry import _synthetic_scan, needs_ocr
from tests.test_text_compare import _text_pdf, needs_font


def _docx(paragraphs: list[str]) -> bytes:
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class _FakeLLM:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    def complete_text(self, messages, json_mode=False):
        self.calls += 1
        return self.answer


def _identity(lines: list[str], llm=None) -> di.Identity:
    identity = di.Identity(filename="x")
    di.classify_lines(identity, lines, llm)
    return identity


# ------------------------------ правила ---------------------------------------


@pytest.mark.parametrize("lines, kind", [
    (["ООО «Ромашка»", "ДОГОВОР ПОСТАВКИ № 15/2026", "г. Энск"], "Договор"),
    (["СЧЁТ-ФАКТУРА № 88 от 02.10.2026"], "Счёт-фактура"),
    (["Счёт на оплату № 41"], "Счёт"),
    (["Дополнительное соглашение № 2", "к договору № 15/2026"], "Дополнительное соглашение"),
    (["ПРОТОКОЛ", "совещания по графику поставки"], "Протокол"),
    (["Служебная записка"], "Служебная записка"),
])
def test_kind_by_title(lines, kind):
    found = _identity(lines)
    assert found.kind == kind and found.basis == "по заголовку на первой странице"


def test_title_must_start_with_capital_and_be_short():
    # строчная — продолжение текста; длинная строка — абзац, а не заголовок
    assert di.find_title(["договор считается заключённым с момента подписания"]) is None
    assert di.find_title(["Акт " + "слово " * 20]) is None


def test_number_and_date_near_title():
    found = _identity(["АКТ № 7/2", "освидетельствования скрытых работ", "от «12» октября 2026 г."])
    assert (found.number, found.date) == ("7/2", "12 октября 2026")


@pytest.mark.parametrize("code_line, kind", [
    ("Унифицированная форма № КС­2", "Акт о приёмке выполненных работ (форма КС-2)"),  # мягкий перенос вместо дефиса
    ("Форма по ОКУД 0322005 КС-3", "Справка о стоимости выполненных работ (форма КС-3)"),
    ("Унифицированная форма № ТОРГ‑12", "Товарная накладная (форма ТОРГ-12)"),
])
def test_unified_form_by_code(code_line, kind):
    found = _identity([code_line, "АКТ О ПРИЁМКЕ ВЫПОЛНЕННЫХ РАБОТ"])
    assert found.kind == kind and found.basis == "по коду унифицированной формы"


def test_letter_by_requisites():
    found = _identity(["ООО «Ромашка»", "Исх. № 12-345 от 03.09.2026", "Уважаемый Иван Петрович!", "Сообщаем…"])
    assert found.kind == "Письмо" and (found.number, found.date) == ("12-345", "03.09.2026")


STAMP = ["Изм.", "Кол.уч", "Лист", "№ док.", "Подп.", "Дата", "Стадия", "Листов", "Разраб.", "Н.контр."]


def test_drawing_by_title_block():
    assert _identity(["Разрез 1-1", *STAMP]).kind == "Чертёж (лист с основной надписью)"


def test_title_wins_over_title_block_but_block_is_mentioned():
    # основная надпись есть и у ведомостей, и у спецификаций — вид по заголовку точнее
    found = _identity(["ВЕДОМОСТЬ ОБЪЁМОВ РАБОТ", *STAMP])
    assert found.kind == "Ведомость" and "лист с основной надписью проектной документации" in found.details


def test_model_is_asked_only_without_title_and_marked_as_estimate():
    llm = _FakeLLM(json.dumps({"вид": "акт осмотра"}))
    found = _identity(["Выезд на площадку 14 октября: ограждение установлено, замечаний нет."], llm)
    assert found.kind == "Акт осмотра" and "оценка модели" in found.basis
    assert _identity(["ДОГОВОР № 1"], llm).kind == "Договор" and llm.calls == 1  # с заголовком модель не нужна


@pytest.mark.parametrize("answer", ["не JSON", json.dumps({"вид": "<script>"}), json.dumps({"вид": ""})])
def test_bad_model_answer_means_unknown(answer):
    assert _identity(["Текст без заголовка о выезде на площадку."], _FakeLLM(answer)).kind == ""


# --------------------- кириллица из САПР без кодировки -------------------------


def test_repair_cp1251_mojibake_word_by_word():
    broken = "Изм. Кол.уч Лист".encode("cp1251").decode("cp1252")  # как отдаёт текстовый слой чертежа
    assert broken != "Изм. Кол.уч Лист"
    assert repair_cp1251(f"{broken} | Подпись и дата") == "Изм. Кол.уч Лист | Подпись и дата"


@pytest.mark.parametrize("text", ["Труба 530×8,0", "café Citroën", "Насос-дозатор НД-25"])
def test_repair_cp1251_keeps_real_text(text):
    assert repair_cp1251(text) == text


def test_huge_sheet_is_not_rendered_for_form_search():
    import pymupdf

    from app.services import spec_registry as sr

    pdf = pymupdf.open()
    page = pdf.new_page(width=1682 / 25.4 * 72, height=594 / 25.4 * 72)  # чертёж в длину
    assert sr.too_large_for_form(page)
    assert not sr.too_large_for_form(pdf.new_page(width=842, height=595))  # A4


# ------------------------------ файлы и чат ------------------------------------


@needs_font
def test_text_pdf_act():
    found = di.identify(_text_pdf(["АКТ № 7", "освидетельствования скрытых работ", "от 12.10.2026"]), "act.pdf", None)
    assert (found.kind, found.number, found.date) == ("Акт", "7", "12.10.2026")
    assert [s.label for s in found.suggestions] == ["Перевести в Word", "Перевести в Excel", "Извлечь реквизиты",
                                                    "Сделать выжимку"]


@needs_ocr
def test_spec_form_is_recognized_by_geometry():
    found = di.identify(_synthetic_scan(), "spec.pdf", None)
    assert found.kind == "Спецификация (форма ГОСТ 21.110)"
    assert found.suggestions[0].message == "составь реестр спецификаций"


@pytest.mark.parametrize("message", ["", "что это?", "Посмотри, что за документ"])
def test_chat_file_without_instructions_is_identified(message):
    files = [("contract.docx", _docx(["ДОГОВОР ПОСТАВКИ № 15/2026", "г. Энск «01» октября 2026 г.", "Текст."]))]
    result = cd.handle_chat_files(message, files, llm=None)
    assert "похоже на: **Договор**" in result.answer and "№ 15/2026" in result.answer
    assert [s["message"] for s in result.suggestions] == ["сделай выжимку", "покажи содержимое"]
    assert result.attachment is None


def test_chat_explicit_request_is_not_intercepted():
    class StubLLM:
        def summarize_text(self, text):
            return "КРАТКО"

    files = [("contract.docx", _docx(["ДОГОВОР ПОСТАВКИ № 15/2026", "Текст договора."]))]
    result = cd.handle_chat_files("сделай выжимку", files, StubLLM())
    assert result.answer.startswith("**Выжимка") and result.suggestions == []


@needs_font
def test_chat_pdf_summary():
    class StubLLM:
        def summarize_text(self, text):
            return "КРАТКО: " + text[:10]

    files = [("act.pdf", _text_pdf(["АКТ № 7", "Работы выполнены в соответствии с проектом."]))]
    result = cd.handle_chat_files("сделай выжимку", files, StubLLM())
    assert result.answer.startswith("**Выжимка — act.pdf:**") and "КРАТКО" in result.answer


def test_api_returns_suggestions(client):
    from tests.conftest import register

    headers = register(client, "identify@test.local")
    files = [("files", ("contract.docx", _docx(["ДОГОВОР № 3", "Текст."]),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))]
    resp = client.post("/api/ask-files", data={"message": ""}, files=files, headers=headers)
    assert resp.status_code == 200, resp.text
    assert [s["label"] for s in resp.json()["suggestions"]] == ["Сделать выжимку", "Показать структуру"]
