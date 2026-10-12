"""Проверка обоснованности ответа и цитаты (app/services/grounding.py)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services import grounding as g


@dataclass
class Chunk:
    text: str
    relative_path: str = "ЕКТП_Транспорт/Инструкция.pdf"
    page: int | None = 3
    section: str | None = None


STATUSES = Chunk(
    "Статусы заявки на транспорт:\n"
    "Черновик — статус после создания заявки, до размещения на обработку.\n"
    "На обработке — заявка размещена и ожидает проверки диспетчером.\n"
    "Принята — заявка согласована ответственным."
)
CREATE = Chunk(
    "Создание заявки. Откройте раздел Парк ТС, нажмите кнопку Новая и заполните обязательные поля: "
    "дату поездки, маршрут и количество пассажиров.",
    relative_path="ЕКТП_Транспорт/Общая инструкция.pdf",
    page=7,
)
UNRELATED = Chunk(
    "Порядок проведения периодического медосмотра. Работник проходит осмотр в медицинской организации "
    "один раз в год по направлению работодателя.",
    relative_path="Обучение_и_медосмотр/Медосмотр.docx",
    page=1,
)
CONTEXT = [STATUSES, CREATE, UNRELATED]


def check(answer: str, chunks=None):
    return g.check_grounding(answer, chunks or CONTEXT)


def test_correct_answer_is_fully_supported_with_citations():
    r = check(
        "Заявка получает статус **Черновик** после создания.\n"
        "- Затем заявка размещается на обработку и ожидает проверки диспетчером.\n"
        "- После согласования ответственным заявка принимается."
    )
    assert r.total == 3 and r.supported == 3 and r.ratio == 1.0
    assert r.unsupported == []
    first = r.citations[0]
    assert first.file_name == "Инструкция.pdf" and first.page == 3
    assert "Черновик" in first.quote or "обработк" in first.quote


def test_quote_is_verbatim_from_the_source_chunk():
    r = check("Заявка на транспорт создаётся в разделе Парк ТС кнопкой Новая.")
    assert r.supported == 1
    quote = r.citations[0].quote
    assert quote in CREATE.text.replace("\n", " ") or quote in CREATE.text
    assert r.citations[0].source.endswith("Общая инструкция.pdf")


def test_invented_number_is_flagged():
    r = check("Срок рассмотрения заявки составляет 14 рабочих дней.")
    claim = r.claims[0]
    assert not claim.supported and claim.missing_numbers == ["14"]
    assert r.citations == []


def test_invented_content_without_numbers_is_flagged():
    r = check("Результат автоматически направляется в бухгалтерию для начисления премии.")
    assert r.supported == 0 and r.total == 1


def test_inflected_forms_are_matched():
    # «заявки / заявка / заявку», «диспетчером / диспетчер»
    r = check("Диспетчер проверяет размещённую заявку.")
    assert r.supported == 1


def test_numbers_present_in_context_are_fine():
    chunks = [Chunk("Срок рассмотрения заявки составляет 14 рабочих дней с даты подачи.")]
    r = check("Срок рассмотрения заявки — 14 рабочих дней.", chunks)
    assert r.supported == 1


def test_single_digit_numbering_is_ignored():
    r = check("Шаг 3: откройте раздел Парк ТС и нажмите Новая.")
    assert r.claims[0].missing_numbers == [] and r.claims[0].supported


def test_three_zones_supported_partial_unsupported():
    chunks = [Chunk("альфа бета гамма дельта")]
    assert check("Альфа бета гамма эпсилон.", chunks).claims[0].status == g.STATUS_SUPPORTED  # 3/4
    assert check("Альфа бета эпсилон дзета.", chunks).claims[0].status == g.STATUS_PARTIAL  # 2/4
    assert check("Альфа эпсилон дзета тета.", chunks).claims[0].status == g.STATUS_UNSUPPORTED  # 1/4


def test_partial_claims_do_not_raise_alarm():
    r = check("Альфа бета эпсилон дзета.", [Chunk("альфа бета гамма дельта")])
    assert r.partial == 1 and r.unsupported == [] and r.ratio == 1.0
    assert r.citations == []  # цитата нужна только для уверенно подтверждённых


def test_meta_phrases_and_list_leadins_are_not_claims():
    claims = g.split_claims(
        "Перечень статусов из контекста:\n"
        "- В базе нет информации о сроках рассмотрения.\n"
        "- Если требуется уточнить статус, проверьте его в системе.\n"
        "- Примечание: список может отличаться.\n"
        "- Заявка размещена и ожидает проверки диспетчером."
    )
    assert claims == ["Заявка размещена и ожидает проверки диспетчером."]


def test_markdown_links_keep_text_but_drop_paths():
    claims = g.split_claims("[Подача заявки на транспорт](ЕКТП_Транспорт/инструкции/Подача.md) – процесс создания заявки.")
    assert claims and "Подача.md" not in claims[0] and "инструкции" not in claims[0]


def test_source_notes_in_parentheses_are_dropped():
    claims = g.split_claims("Заявка размещена и ожидает проверки диспетчером (из документа Инструкция.pdf, раздел 4).")
    assert claims and "Инструкция.pdf" not in claims[0]


def test_headings_and_short_lines_are_not_claims():
    claims = g.split_claims("**Статусы заявки:**\n\n1. Черновик\n- ОК\nЗаявка размещена и ожидает проверки диспетчером.")
    assert claims == ["Заявка размещена и ожидает проверки диспетчером."]


def test_markdown_is_stripped_from_claims():
    claims = g.split_claims("- **Черновик** – статус после создания заявки, до размещения на обработку.")
    assert claims and "*" not in claims[0] and not claims[0].startswith("-")


def test_empty_answer():
    r = check("")
    assert r.total == 0 and r.ratio == 1.0 and r.citations == []


def test_citations_are_deduplicated_and_ordered_by_support():
    r = check(
        "Заявка получает статус Черновик после создания, до размещения на обработку.\n"
        "Черновик — это статус после создания заявки, до размещения на обработку.\n"
        "В разделе Парк ТС нажмите кнопку Новая и заполните обязательные поля."
    )
    keys = [(c.source, c.quote) for c in r.citations]
    assert len(keys) == len(set(keys))
    assert len(r.citations[0].claims) >= len(r.citations[-1].claims)
    assert len(r.citations) <= g.MAX_CITATIONS


def test_summary_is_json_serializable():
    import json

    r = check("Срок составляет 14 рабочих дней по заявке на транспорт.\nЗаявка размещена и ожидает проверки диспетчером.")
    data = json.loads(json.dumps(r.summary(), ensure_ascii=False))
    assert set(data) == {"supported", "partial", "total", "ratio", "unsupported", "citations"}
    assert data["total"] == 2 and data["supported"] == 1
    assert data["unsupported"][0].startswith("Срок")


@pytest.mark.parametrize("number, text, expected", [
    ("14", "срок 14 дней", True),
    ("14", "номер 114", False),
    ("1.5", "ставка 1,5 процента", True),
    ("20", "НДС 20%", True),
])
def test_number_matching(number, text, expected):
    assert g._has_number(number, text) is expected


def test_quote_is_cleaned_of_markdown_markup():
    chunk = Chunk("- `Назначить` ответственного -> `Поручена`. Остальной текст про другое.")
    r = g.check_grounding("Назначить ответственного — статус Поручена.", [chunk])
    quote = r.citations[0].quote
    assert "`" not in quote and not quote.startswith("-") and "Назначить ответственного" in quote
