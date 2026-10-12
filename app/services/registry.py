"""Пакетный разбор документов в реестр (Excel): PDF и сканы → таблица с заданными полями.

Сценарий: пользователь прикладывает несколько PDF (акты, письма, счета) и пишет, какие
поля нужны, — получает один .xlsx, строка на документ.

Главное свойство — недоверие к модели. Локальная модель может выдать поле, которого в
документе нет. Поэтому каждое извлечённое значение проверяется по тексту документа;
не найденное там подсвечивается в Excel и попадает в колонку «Проверить». Текст сканов
получен OCR, поэтому такие строки тоже помечены.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field

from app.services.document_processing import DocumentError, PdfData, append_row

DEFAULT_FIELDS = ["Тип документа", "Номер", "Дата", "Контрагент", "Сумма", "Краткое содержание"]
# Подсказки модели для стандартных полей (для пользовательских полей их нет).
FIELD_HINTS = {
    "Тип документа": "вид документа одним-двумя словами: акт, счёт, письмо, договор, накладная, протокол и т.п.",
    "Номер": "номер документа (регистрационный, исходящий, номер акта или счёта)",
    "Дата": "дата документа, как написана в тексте",
    "Контрагент": "организация-контрагент: поставщик, подрядчик или отправитель; название одной организации",
    "Сумма": "итоговая сумма документа, как написана в тексте, только число без валюты",
    "Краткое содержание": "суть документа одним предложением",
}
MAX_REGISTRY_DOCS = 10  # на локальной модели ~30–60 с на документ: больше — ждать слишком долго
MAX_REGISTRY_FIELDS = 12
MAX_FIELD_LEN = 60
TEXT_HEAD_CHARS = 4500  # реквизиты обычно в начале документа
TEXT_TAIL_CHARS = 1500  # итоговые суммы и подписи — в конце

# Поля, чьё значение модель пишет своими словами: сверять с текстом бессмысленно.
_FREE_TEXT_FIELD_RE = re.compile(
    r"содержан|суть|описан|тема|предмет|комментар|назначен|резюме|кратк|примечан|заключен",
    re.IGNORECASE,
)

_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_DATE_RE = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$")


@dataclass
class RegistryRow:
    filename: str
    values: dict[str, str]  # поле → значение ("" если не найдено)
    unverified: list[str] = field(default_factory=list)  # поля, которых нет в тексте документа
    pages: int = 0
    scanned: bool = False  # хотя бы одна страница распознана OCR
    error: str | None = None  # документ не удалось разобрать


@dataclass
class RegistryResult:
    fields: list[str]
    rows: list[RegistryRow]
    seconds: float
    default_fields_used: bool

    @property
    def filled_cells(self) -> int:
        return sum(1 for r in self.rows for v in r.values.values() if v)

    @property
    def unverified_cells(self) -> int:
        return sum(len(r.unverified) for r in self.rows)


# ----------------------------- поля из сообщения -------------------------


def parse_fields_from_message(message: str) -> list[str] | None:
    """Список полей из текста запроса или None, если пользователь их не назвал.

    Понимает «реестр: номер, дата, сумма», «поля: …», «извлеки номер, дату и сумму».
    """
    text = (message or "").strip()
    if not text:
        return None

    tail: str | None = None
    m = re.search(r"(?:поля|колонки|столбцы|реквизиты)\s*:?\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    if m:
        tail = m.group(1)
    elif ":" in text:
        tail = text.split(":", 1)[1]
    else:
        m = re.search(
            r"(?:извлеки|извлечь|выпиши|вытащи|достань|собери)\s+(.+)$", text, re.IGNORECASE | re.DOTALL
        )
        if m and re.search(r"[,;]| и ", m.group(1)):
            tail = m.group(1)
    if not tail:
        return None

    # «…сумму из каждого файла», «…в Excel» — это не часть названия поля
    tail = re.sub(
        r"\s+(?:из|по|для)\s+(?:каждого|каждому|всех|этих|данных|приложенных|файл\w*|документ\w*)\b.*$",
        "",
        tail,
        flags=re.IGNORECASE | re.DOTALL,
    )
    tail = re.sub(r"\s+в\s+(?:excel|эксель|xlsx|таблицу)\b.*$", "", tail, flags=re.IGNORECASE | re.DOTALL)
    parts = re.split(r"[,;\n]+|\s+и\s+", tail)
    fields: list[str] = []
    seen: set[str] = set()
    for part in parts:
        name = part.strip().strip("«»\"'.•-– ").strip()
        # «из каждого файла», «в excel» и прочие хвосты — не поля
        if not name or len(name) > MAX_FIELD_LEN or len(name.split()) > 5:
            continue
        key = name.lower()
        if key in seen or key in {"excel", "эксель", "xlsx", "пожалуйста"}:
            continue
        seen.add(key)
        fields.append(name[0].upper() + name[1:])
    return fields[:MAX_REGISTRY_FIELDS] or None


# --------------------------- проверка по тексту --------------------------


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower().replace("ё", "е")).split())


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def is_free_text_field(name: str) -> bool:
    return bool(_FREE_TEXT_FIELD_RE.search(name))


def is_grounded(value: str, doc_text: str) -> bool:
    """Есть ли значение в тексте документа (с поправкой на форматы дат, чисел, пробелов)."""
    if not value.strip():
        return True
    norm_text = _norm(doc_text)
    norm_value = _norm(value)
    if norm_value and norm_value in norm_text:
        return True

    # Даты: «15.11.2026» ≈ «15 ноября 2026 г.»
    dm = _DATE_RE.match(value.strip())
    if dm:
        day, month, year = int(dm.group(1)), int(dm.group(2)), dm.group(3)
        year = year if len(year) == 4 else "20" + year
        month_name = next((n for n, i in _MONTHS.items() if i == month), "")
        day_ok = re.search(rf"(?<!\d)0?{day}(?!\d)", norm_text) is not None
        month_ok = month_name in norm_text or re.search(rf"(?<!\d)0?{month}(?!\d)", norm_text) is not None
        return day_ok and month_ok and year in norm_text

    # Числа и суммы: «1 250 000,00» ≈ «1250000» — сравниваем по цифрам.
    value_digits = _digits(value)
    if len(value_digits) >= 3 and not re.search(r"[a-zа-я]{3,}", norm_value):
        return value_digits in _digits(doc_text)

    # Слова (организации, ФИО, виды документов): все значимые слова значения есть в тексте.
    # Корень слова (первые 5 букв) терпит падежи: «ООО Вектор» ≈ «ООО "Вектор"».
    words = [w for w in norm_value.split() if len(w) >= 3 or w.isdigit()]
    if not words:
        return False
    text_words = norm_text.split()
    stems = {w[:5] for w in text_words}
    return all(w[:5] in stems for w in words)


# ------------------------------ извлечение -------------------------------


def _document_text(data: PdfData) -> str:
    """Текст страниц плюс строки таблиц (реквизиты и суммы часто лежат именно там)."""
    parts = [data.full_text]
    for table in data.tables:
        for row in table.rows:
            parts.append(" | ".join(c for c in row if c))
    return "\n".join(p for p in parts if p.strip())


def _llm_view(doc_text: str) -> str:
    if len(doc_text) <= TEXT_HEAD_CHARS + TEXT_TAIL_CHARS:
        return doc_text
    return doc_text[:TEXT_HEAD_CHARS] + "\n[…]\n" + doc_text[-TEXT_TAIL_CHARS:]


_NUMBER_RE = re.compile(
    r"\s*(\d[\d\s\u00a0.,]*\d|\d)\s*(?:руб\w*\.?|₽|р\.|rub|usd|eur|%)?\s*$", re.IGNORECASE
)
_NUMBER_TOKEN_RE = re.compile(r"\d[\d\u00a0 ]*(?:[.,]\d+)?")


def _to_float(text: str) -> float | None:
    cleaned = re.sub(r"[\s\u00a0]", "", text).replace(",", ".")
    if cleaned.count(".") > 1:  # «1.250.000» — точки как разделители тысяч
        head, _, tail = cleaned.rpartition(".")
        cleaned = head.replace(".", "") + "." + tail
    try:
        return float(cleaned)
    except ValueError:
        return None


def snap_number_to_source(value: str, doc_text: str) -> str:
    """Вернуть число в исходной записи документа («412800.0» → «412 800,00»).

    Модель нередко «нормализует» суммы. В реестре нужна запись как в документе, а не
    пересчитанная: сравниваем значения числами и берём дословный фрагмент из текста.
    Если однозначного совпадения нет — оставляем как есть.
    """
    m = _NUMBER_RE.match(value)
    if not m:
        return value
    target = _to_float(m.group(1))
    if target is None:
        return value
    seen: list[str] = []
    for token in _NUMBER_TOKEN_RE.findall(doc_text):
        token = token.strip()
        if token not in seen and _to_float(token) == target:
            seen.append(token)
    # если запись неоднозначна (разные написания одного числа), берём первое — по порядку в тексте
    return seen[0] if seen else value


def _lookup(extracted: dict, name: str) -> str:
    """Значение поля из ответа модели; ключ ищем без учёта регистра и пробелов."""
    if name in extracted and extracted[name]:
        return str(extracted[name]).strip()
    wanted = _norm(name)
    for key, value in extracted.items():
        if _norm(str(key)) == wanted and value:
            return str(value).strip()
    return ""


def extract_row(data: PdfData, fields: list[str], llm) -> RegistryRow:
    row = RegistryRow(
        filename=data.filename,
        values={f: "" for f in fields},
        pages=len(data.pages),
        scanned=bool(data.scanned_pages) and data.ocr_available,
    )
    doc_text = _document_text(data)
    if not doc_text.strip():
        row.error = (
            "скан без распознавания (OCR недоступен)"
            if data.scanned_pages and not data.ocr_available
            else "в документе не найден текст"
        )
        return row

    spec = [{"key": f, "title": f, "description": FIELD_HINTS.get(f)} for f in fields]
    extracted = llm.extract_fields(_llm_view(doc_text), spec)  # RuntimeError → выше (503)

    for name in fields:
        value = _lookup(extracted, name)
        if value and re.search(r"номер|№", name, re.IGNORECASE):
            value = re.sub(r"^(?:№|N|No\.?|Номер)\s*", "", value).strip()  # «№ 47» → «47»
        if value and not is_free_text_field(name):
            value = snap_number_to_source(value, doc_text)
        row.values[name] = value
        if value and not is_free_text_field(name) and not is_grounded(value, doc_text):
            row.unverified.append(name)
    return row


def build_registry(docs: list[PdfData], fields: list[str] | None, llm) -> RegistryResult:
    if not docs:
        raise DocumentError("Не приложено ни одного PDF")
    if len(docs) > MAX_REGISTRY_DOCS:
        raise DocumentError(
            f"Слишком много документов для одного реестра ({len(docs)}). "
            f"Лимит — {MAX_REGISTRY_DOCS}: модель работает локально и разбирает документ за десятки секунд. "
            "Разбейте на несколько запросов."
        )
    use_defaults = not fields
    columns = list(DEFAULT_FIELDS if use_defaults else fields)

    started = time.monotonic()
    rows = [extract_row(d, columns, llm) for d in docs]
    return RegistryResult(
        fields=columns, rows=rows, seconds=time.monotonic() - started, default_fields_used=use_defaults
    )


# --------------------------------- Excel ---------------------------------


def build_registry_xlsx(result: RegistryResult) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Реестр"

    headers = ["Файл", *result.fields, "Проверить", "Стр.", "OCR"]
    append_row(ws, headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        cell.fill = PatternFill("solid", fgColor="E7EEF8")

    warn = PatternFill("solid", fgColor="FFF2B3")  # значение не найдено в тексте документа
    bad = Font(color="B00020")

    for row in result.rows:
        if row.error:
            append_row(ws, [row.filename, *([""] * len(result.fields)), f"НЕ РАЗОБРАН: {row.error}", row.pages, ""])
            ws.cell(ws.max_row, 1).font = bad
            ws.cell(ws.max_row, len(result.fields) + 2).font = bad
            continue
        append_row(
            ws,
            [
                row.filename,
                *[row.values[f] for f in result.fields],
                ", ".join(row.unverified),
                row.pages,
                "да" if row.scanned else "",
            ],
        )
        r = ws.max_row
        for i, name in enumerate(result.fields, start=2):
            ws.cell(r, i).alignment = Alignment(vertical="top", wrap_text=True)
            if name in row.unverified:
                ws.cell(r, i).fill = warn
        if row.scanned:
            ws.cell(r, len(headers)).fill = warn

    ws.freeze_panes = "B2"
    for i, header in enumerate(headers, start=1):
        longest = max((len(str(c.value or "")) for c in ws[get_column_letter(i)]), default=10)
        ws.column_dimensions[get_column_letter(i)].width = min(max(12, longest + 2), 48)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
