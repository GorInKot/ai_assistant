"""Что за документ и что с ним можно сделать — когда файл приложен без указаний.

Только подсказка: ассистент ничего не делает сам, а называет вид документа, показывает,
на чём основан вывод, и предлагает действия (в интерфейсе — кнопками). Поэтому ошибка
в определении вида безобидна: пользователь просто выберет нужное действие.

Как определяется вид (по убыванию надёжности):
1. Форма бланка: таблица спецификации ГОСТ 21.110 — по геометрии граф, без модели.
2. Заголовок первой страницы: деловые документы сами называют себя («АКТ о приёмке…»,
   «ДОГОВОР ПОСТАВКИ № 15», «Счёт-фактура»); унифицированные формы — по коду (КС-2, ТОРГ-12);
   письма — по реквизитам («Исх. №», «Уважаемый…»); листы проектной документации — по
   основной надписи («Изм.», «Кол.уч.», «№ док.», «Подп.»). Рядом с заголовком ищутся номер и дата.
3. Если ничего из этого нет — вид оценивает локальная модель по началу текста, и это прямо
   помечено: «оценка модели, может ошибаться».
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

TITLE_LINES = 25  # заголовок ищем в первых строках первой страницы
TITLE_MAX_WORDS = 15  # длиннее — это уже текст, а не заголовок
LLM_HEAD_CHARS = 1500  # сколько начала текста показываем модели
STAMP_MIN_WORDS = 4  # столько слов основной надписи — лист проектной документации


@dataclass
class Suggestion:
    label: str  # подпись кнопки
    message: str  # что отправить вместе с тем же файлом


@dataclass
class Identity:
    filename: str
    kind: str = ""  # «Акт», «Спецификация (форма ГОСТ 21.110)», «Письмо»; пусто — не определён
    basis: str = ""  # на чём основан вывод
    title: str = ""  # заголовок как в документе
    number: str = ""
    date: str = ""
    details: list[str] = field(default_factory=list)  # «3 стр.», «скан — текст распознан»
    suggestions: list[Suggestion] = field(default_factory=list)
    hint: str = ""  # что ещё можно сделать (требует другого файла)


# Вид документа по первому слову заголовка: (шаблон начала строки, как назвать). Порядок важен:
# «счёт-фактура» раньше «счёт», «дополнительное соглашение» раньше «соглашение».
_KINDS = [
    (r"универсальн\w* передаточн\w* документ|упд\b", "Универсальный передаточный документ"),
    (r"сч[её]т[-\s]?фактур", "Счёт-фактура"),
    (r"сч[её]т\b", "Счёт"),
    (r"товарн\w* накладн", "Товарная накладная"),
    (r"транспортн\w* накладн", "Транспортная накладная"),
    (r"накладн", "Накладная"),
    (r"дополнительн\w* соглашени", "Дополнительное соглашение"),
    (r"соглашени", "Соглашение"),
    (r"договор|контракт", "Договор"),
    (r"акт\b", "Акт"),
    (r"служебн\w* записк", "Служебная записка"),
    (r"докладн\w* записк", "Докладная записка"),
    (r"пояснительн\w* записк", "Пояснительная записка"),
    (r"приказ", "Приказ"),
    (r"распоряжени", "Распоряжение"),
    (r"протокол", "Протокол"),
    (r"предписани", "Предписание"),
    (r"уведомлени", "Уведомление"),
    (r"претензи", "Претензия"),
    (r"заявк", "Заявка"),
    (r"заявлени", "Заявление"),
    (r"справк", "Справка"),
    (r"доверенност", "Доверенность"),
    (r"техническ\w* задани", "Техническое задание"),
    (r"техническ\w* услови", "Технические условия"),
    (r"спецификаци", "Спецификация"),
    (r"ведомост", "Ведомость"),
    (r"локальн\w* сметн|сметн\w* расч|смета", "Смета"),
    (r"журнал", "Журнал"),
    (r"отч[её]т", "Отчёт"),
    (r"паспорт", "Паспорт"),
    (r"сертификат", "Сертификат"),
    (r"декларац", "Декларация"),
    (r"инструкци", "Инструкция"),
    (r"положени", "Положение"),
    (r"регламент", "Регламент"),
    (r"стандарт", "Стандарт"),
    (r"методик", "Методика"),
    (r"программ", "Программа"),
    (r"график", "График"),
    (r"план\b", "План"),
    (r"перечень", "Перечень"),
    (r"реестр", "Реестр"),
    (r"заключени", "Заключение"),
    (r"решени", "Решение"),
    (r"наряд[-\s]?допуск", "Наряд-допуск"),
    (r"исполнительн\w* схем", "Исполнительная схема"),
    (r"письм", "Письмо"),
]
_KIND_RES = [(re.compile(r"^[\s\"«„(]*(?:" + pattern + r")", re.IGNORECASE), name) for pattern, name in _KINDS]

# Унифицированные формы — по коду формы где угодно в начале документа. Дефис в PDF бывает
# любым: обычный, мягкий перенос (U+00AD), неразрывный, короткое тире.
_DASH = "[-\u00ad\u2010\u2011\u2013\\s]?"
_FORM_CODES = [
    (re.compile(rf"\bКС{_DASH}2\b", re.IGNORECASE), "Акт о приёмке выполненных работ (форма КС-2)"),
    (re.compile(rf"\bКС{_DASH}3\b", re.IGNORECASE), "Справка о стоимости выполненных работ (форма КС-3)"),
    (re.compile(rf"\bТОРГ{_DASH}12\b", re.IGNORECASE), "Товарная накладная (форма ТОРГ-12)"),
    (re.compile(rf"\bМ{_DASH}29\b"), "Отчёт о расходе материалов (форма М-29)"),
]
_LETTER_RE = re.compile(r"\bисх(?:одящий)?\.?\s*№|\bуважаем\w+\b", re.IGNORECASE)
_OUTGOING_RE = re.compile(r"\bисх(?:одящий)?\.?\s*№", re.IGNORECASE)
_STAMP_WORDS = ("изм.", "кол.уч", "№ док", "подп.", "дата", "стадия", "листов", "разраб", "пров.", "н.контр", "гип")
_NUMBER_RE = re.compile(r"№\s*([\w][\w./\-]*)")
_DATE_RE = re.compile(
    r"\b(\d{1,2}[./]\d{1,2}[./](?:\d{4}|\d{2}))\b"
    r"|«?\s*(\d{1,2})\s*»?\s+(январ\w*|феврал\w*|март\w*|апрел\w*|ма[яй]\w*|июн\w*|июл\w*|август\w*|"
    r"сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\s+(\d{4})",
    re.IGNORECASE,
)


def find_title(lines: list[str]) -> tuple[str, str, int] | None:
    """Заголовок с видом документа: (вид, строка заголовка, её индекс) или None."""
    for index, line in enumerate(lines[:TITLE_LINES]):
        text = " ".join(line.split())
        if not text or len(text.split()) > TITLE_MAX_WORDS:
            continue
        first = next((ch for ch in text if ch.isalpha()), "")
        if not first.isupper():  # заголовок пишут с заглавной; строчная — продолжение текста
            continue
        for pattern, kind in _KIND_RES:
            if pattern.match(text):
                return kind, text, index
    return None


def find_form_code(lines: list[str]) -> str:
    head = "\n".join(lines[:TITLE_LINES * 2])
    return next((kind for pattern, kind in _FORM_CODES if pattern.search(head)), "")


def is_letter(lines: list[str]) -> bool:
    return any(_LETTER_RE.search(line) for line in lines[:TITLE_LINES * 2])


def is_drawing_sheet(text: str) -> bool:
    """Основная надпись листа проектной документации («Изм.», «Кол.уч.», «№ док.», «Подп.»)."""
    lower = text.lower()
    return sum(1 for word in _STAMP_WORDS if word in lower) >= STAMP_MIN_WORDS


def number_and_date(lines: list[str], start: int) -> tuple[str, str]:
    """Номер и дата рядом с заголовком (в нём и в трёх следующих строках)."""
    near = " ".join(lines[start:start + 4])
    number = _NUMBER_RE.search(near)
    date = _DATE_RE.search(near)
    date_text = ""
    if date:
        date_text = date.group(1) or f"{date.group(2)} {date.group(3)} {date.group(4)}"
    return (number.group(1).rstrip(".,") if number else ""), date_text


def ask_model(text: str, llm) -> str:
    """Вид документа по началу текста — оценка модели (пусто, если модели нет или ответ негоден)."""
    if llm is None or not text.strip():
        return ""
    prompt = (
        "Определи вид документа по его началу (акт, письмо, договор, протокол, отчёт, инструкция, "
        "счёт и т. п.). Ответь JSON: {\"вид\": \"одно-три слова по-русски\"}. "
        "Если понять нельзя — {\"вид\": \"\"}. Ничего не придумывай."
    )
    try:
        answer = llm.complete_text(
            [{"role": "system", "content": prompt}, {"role": "user", "content": text[:LLM_HEAD_CHARS]}],
            json_mode=True,
        )
        kind = str(json.loads(answer).get("вид", "")).strip()
    except Exception:  # noqa: BLE001 — модель недоступна или ответ не JSON: вид не определён
        return ""
    if not re.fullmatch(r"[А-Яа-яЁё][А-Яа-яЁё\- ]{1,39}", kind):
        return ""
    return kind[:1].upper() + kind[1:]


def classify_lines(identity: Identity, lines: list[str], llm) -> None:
    """Вид документа по строкам первой страницы: код формы, заголовок, письмо, основная надпись, модель."""
    text = "\n".join(lines)
    form = find_form_code(lines)
    title = find_title(lines)
    if title:
        identity.title = title[1]
        identity.number, identity.date = number_and_date(lines, title[2])
    if form:
        identity.kind, identity.basis = form, "по коду унифицированной формы"
    elif title:
        identity.kind, identity.basis = title[0], "по заголовку на первой странице"
    elif is_letter(lines):
        identity.kind, identity.basis = "Письмо", "по реквизитам письма: исходящий номер, обращение"
        outgoing = next((i for i, line in enumerate(lines[:TITLE_LINES]) if _OUTGOING_RE.search(line)), None)
        if outgoing is not None:
            identity.number, identity.date = number_and_date(lines, outgoing)
    elif is_drawing_sheet(text):
        identity.kind = "Чертёж (лист с основной надписью)"
        identity.basis = "по основной надписи (штампу)"
    if identity.kind and identity.kind != "Чертёж (лист с основной надписью)" and is_drawing_sheet(text):
        # основная надпись есть у любого документа проекта (спецификации, ведомости) — вид по заголовку точнее
        identity.details.append("лист с основной надписью проектной документации")
    if not identity.kind:
        kind = ask_model(text, llm)
        if kind:
            identity.kind, identity.basis = kind, "по содержанию — оценка модели, может ошибаться"


# --------------------------------- файлы ---------------------------------------

_COMPARE_HINT = "Чтобы сравнить с другой версией, приложите оба файла и напишите «сравни»."


def identify_pdf(content: bytes, filename: str, llm) -> Identity:
    import pymupdf

    from app.services import spec_registry as sr
    from app.services import text_compare as tc
    from app.services.document_processing import (
        MAX_FILE_SIZE_PDF,
        MIN_TEXT_LAYER_CHARS,
        DocumentError,
        _check_size,
        repair_cp1251,
    )

    _check_size(content, MAX_FILE_SIZE_PDF)
    try:
        pdf = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise DocumentError(f"Не удалось прочитать «{filename}» как PDF: {exc}") from exc
    if pdf.needs_pass:
        raise DocumentError(f"«{filename}» защищён паролем — снимите защиту и загрузите снова")
    identity = Identity(filename=filename, details=[f"{pdf.page_count} стр."])
    page = pdf[0]
    text_layer = (page.get_text() or "").strip()
    scanned = len(text_layer) < MIN_TEXT_LAYER_CHARS and bool(page.get_images())

    spec = False
    if sr.unavailable_reason() is None:
        for index in range(min(3, pdf.page_count)):  # первые страницы бывают титулом и содержанием
            if sr.too_large_for_form(pdf[index]):
                continue
            gray, colored = sr.render_page(pdf[index])
            try:
                sr.locate_form(gray, colored)
            except sr.SpecFormNotFound:
                continue
            except Exception:  # noqa: BLE001 — поиск формы best-effort
                break
            spec = True
            break
    if scanned:
        ocr = tc._ocr_page(page)
        lines = [" ".join(word for word, _conf in line) for line in (ocr or [])]
        identity.details.append("скан — текст первой страницы распознан" if ocr else
                                "скан, а распознавание на сервере не установлено")
    else:
        text_layer = repair_cp1251(text_layer)  # «Èçì.» вместо «Изм.» — чертёж из САПР
        lines = [ln for ln in text_layer.splitlines() if ln.strip()]
        identity.details.append("PDF с текстовым слоем")
    if spec:
        identity.kind, identity.basis = "Спецификация (форма ГОСТ 21.110)", "по форме таблицы (9 граф бланка)"
        found = find_title(lines)
        identity.title = found[1] if found else ""
        identity.suggestions = [
            Suggestion("Составить реестр позиций", "составь реестр спецификаций"),
            Suggestion("Перевести в Excel", "переведи в Excel"),
        ]
    else:
        classify_lines(identity, lines, llm)
        identity.suggestions = [
            Suggestion("Перевести в Word", "переведи в Word"),
            Suggestion("Перевести в Excel", "переведи в Excel"),
            Suggestion("Извлечь реквизиты", "извлеки реквизиты"),
            Suggestion("Сделать выжимку", "сделай выжимку"),
        ]
    identity.hint = _COMPARE_HINT
    return identity


def identify_word(content: bytes, filename: str, llm) -> Identity:
    from app.services.document_processing import parse_docx

    doc = parse_docx(content, filename)
    identity = Identity(filename=filename, details=[f"абзацев: {len(doc.paragraphs)}, таблиц: {len(doc.tables)}"])
    classify_lines(identity, [p.text for p in doc.paragraphs], llm)
    identity.suggestions = [
        Suggestion("Сделать выжимку", "сделай выжимку"),
        Suggestion("Показать структуру", "покажи содержимое"),
    ]
    identity.hint = _COMPARE_HINT
    return identity


def identify(content: bytes, filename: str, llm) -> Identity:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return identify_pdf(content, filename, llm)
    return identify_word(content, filename, llm)
