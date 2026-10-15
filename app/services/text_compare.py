"""Сравнение двух версий текстового документа по словам: PDF (текстовый слой или скан) и Word.

Почему по словам, а не по строкам: в новой версии текст переверстан, и одно добавленное
слово сдвигает переносы во всём абзаце — построчное сравнение показало бы десятки
«изменённых» строк. Слова обеих версий выстраиваются в порядке чтения и сравниваются
(difflib); различия собираются во фрагменты «было → стало» с контекстом.

Сначала проверяется, что это вообще версии одного документа: какая доля трёхсловий
меньшего документа есть в другом. Иначе сравнение не строится.

Сканы распознаются Tesseract с уверенностью по каждому слову. Как и в реестре
спецификаций, ничего не утверждается наугад: различие, где слово скана прочитано
неуверенно или «было» и «стало» почти совпадают по написанию («дотовор» — «договор»,
«180» — «100»), — не «изменено», а «проверить». Номера страниц и колонтитулы,
повторяющиеся на страницах, в сравнение не входят.
"""

from __future__ import annotations

import io
import re
import shutil
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.services import document_processing as dp
from app.services.document_processing import (
    MAX_DIFF_ITEMS,
    MAX_FILE_SIZE_PDF,
    MAX_FILE_SIZE_WORD,
    MAX_PDF_PAGES,
    MIN_TEXT_LAYER_CHARS,
    OCR_LANG,
    OCR_TIMEOUT,
    DocumentError,
    _check_size,
)

SURE_CONF = 90  # уверенность Tesseract ниже — слово скана прочитано неуверенно
LOOKS_LIKE_OCR = 0.6  # «было» и «стало» на скане похожи сильнее — вероятна ошибка распознавания
SAME_DOC_SHARE = 0.35  # столько трёхсловий меньшего документа есть в другом — версии одного документа
SHINGLE = 3
CONTEXT = 5  # слов контекста по сторонам изменения
MERGE_GAP = 2  # изменения, разделённые не больше чем двумя общими словами, — один фрагмент
MAX_WORDS = 80_000  # слов в документе (~150 страниц текста)
MAX_SCANNED_PAGES = 60  # страниц-сканов на одно сравнение: распознавание ~3–5 с на страницу
SCAN_DPI = 300
FORCED_MAX_CELLS = 200_000_000  # «сравни всё равно» для непохожих документов: предел работы difflib

STATUS_CHANGED = "изменено"
STATUS_ADDED = "добавлено"
STATUS_REMOVED = "удалено"
STATUS_CHECK = "проверить"

_LOOKALIKES = str.maketrans("abcehkmoptxy", "авсенкмортху")


def word_key(text: str) -> str:
    """Слово для сравнения: без регистра и знаков, «1,5» = «1.5», латинские двойники → кириллица."""
    # «№» до нормализации: NFKC превращает его в «No» — получилось бы лишнее «слово»
    t = unicodedata.normalize("NFKC", text.replace("№", "")).lower().replace("ё", "е")
    t = re.sub(r"(?<=\d)[,.](?=\d)", ".", t)
    t = re.sub(r"[^\w.]", "", t).strip("._")
    return t.translate(_LOOKALIKES)


@dataclass
class Word:
    text: str  # как в документе
    key: str  # для сравнения
    where: str  # «стр. 3», «абз. 12», «табл. 2»
    conf: float = 100.0  # для скана — уверенность Tesseract; текстовый слой и Word — 100
    scanned: bool = False


@dataclass
class TextDoc:
    filename: str
    words: list[Word]
    pages: int = 0
    scanned_pages: list[int] = field(default_factory=list)
    dropped_lines: int = 0  # номера страниц и колонтитулы, не вошедшие в сравнение


# ------------------------------ извлечение слов ------------------------------

_Line = list[tuple[str, float]]  # слова строки с уверенностью

_PAGE_NUMBER_RE = re.compile(
    r"^(?:(?:стр(?:аница)?|лист|с)\.?\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?(?:\s*(?:из|/)\s*\d{1,4})?$", re.IGNORECASE
)


def _join_hyphens(lines: list[_Line]) -> list[_Line]:
    """«дого-» в конце строки + «вора» в начале следующей → «договора»."""
    result = [list(line) for line in lines if line]
    for i in range(len(result) - 1):
        line, nxt = result[i], result[i + 1]
        if line and nxt and re.search(r"\w[-‐¬]$", line[-1][0]) and nxt[0][0][:1].islower():
            text, conf = line[-1]
            line[-1] = (text[:-1] + nxt[0][0], min(conf, nxt[0][1]))
            del nxt[0]
    return [line for line in result if line]


def _ocr_page(page) -> list[_Line] | None:
    """Строки скана со словами и уверенностью Tesseract; None — распознавание недоступно."""
    if not shutil.which("tesseract"):
        return None
    try:
        import numpy as np
        import pymupdf
        import pytesseract
        from PIL import Image

        from app.services import spec_registry as sr
    except ImportError:
        return None
    pix = page.get_pixmap(dpi=SCAN_DPI, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width].copy()
    angle = sr.osd_rotation(gray)
    if angle:
        gray = sr._rot90(gray, angle)
    # Линии таблиц (длиннее 12 мм) — белым: иначе Tesseract читает сетку как буквы («ПО ОО»)
    # и теряет строки таблицы целиком. Буквы короче и не задеваются.
    vert, horiz = sr.line_masks(sr.binarize(gray))
    gray = sr.erase_lines(gray, vert | horiz)
    try:
        data = pytesseract.image_to_data(Image.fromarray(gray), lang=OCR_LANG, timeout=OCR_TIMEOUT,
                                         output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001 — сбой одной страницы: она останется пустой, сравнение скажет об этом
        return []
    words: dict[tuple[int, int, int], list[list]] = {}
    for i, text in enumerate(data["text"]):
        if text.strip():
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            words.setdefault(key, []).append([text.strip(), float(data["conf"][i]), box])
    _reread_numbers(gray, [w for line in words.values() for w in line])
    return [[(text, conf) for text, conf, _box in line] for line in _row_order(list(words.values()))]


def _row_order(lines: list[list[list]]) -> list[list[list]]:
    """Строки скана в порядке «сверху вниз, на одной высоте — слева направо».

    Tesseract делит страницу на блоки по-разному в двух сканах одного документа: ячейки
    таблицы он читает то по столбцам, то по строкам, и порядок слов расходится. Положение
    строки на листе от скана к скану не меняется — по нему и выстраиваем."""
    placed = []
    for line in lines:
        top = min(w[2][1] for w in line)
        bottom = max(w[2][1] + w[2][3] for w in line)
        placed.append(((top + bottom) / 2, bottom - top, min(w[2][0] for w in line), line))
    placed.sort(key=lambda item: item[0])
    rows: list[list] = []  # [середина, высота, [(x, строка)]]
    for middle, height, x, line in placed:
        if rows and abs(middle - rows[-1][0]) <= 0.5 * max(height, rows[-1][1]):
            rows[-1][2].append((x, line))
        else:
            rows.append([middle, height, [(x, line)]])
    return [line for _middle, _height, row in rows for _x, line in sorted(row, key=lambda item: item[0])]


REREAD_DIGITS = "--psm 6 -c tessedit_char_whitelist=0123456789.,-/%"
UNSURE_CONF = 40.0


def _reread_numbers(gray, words: list[list]) -> None:
    """Второе прочтение чисел скана: каждое слово с цифрами вырезается, увеличивается вдвое
    и читается ещё раз только цифрами (все числа страницы — одной полосой, один вызов).
    Цифры не сошлись — число прочитано неуверенно: «10» вместо «18» не станет «изменено»."""
    import pytesseract
    from PIL import Image

    numeric = [w for w in words if re.search(r"\d", w[0])]
    if not numeric:
        return
    pad, gap = 6, 30
    crops = []
    for _text, _conf, (x, y, w, h) in numeric:
        crop = Image.fromarray(gray[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad])
        crops.append(crop.resize((crop.width * 2, crop.height * 2)))
    strip = Image.new("L", (max(c.width for c in crops) + 2 * gap, sum(c.height + gap for c in crops) + gap), 255)
    spans, top = [], gap
    for crop in crops:
        strip.paste(crop, (gap, top))
        spans.append((top - gap / 2, top + crop.height + gap / 2))
        top += crop.height + gap
    try:
        data = pytesseract.image_to_data(strip, lang="rus", config=REREAD_DIGITS, timeout=OCR_TIMEOUT,
                                         output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001 — без второго прочтения все числа страницы считаем неуверенными
        data = {"text": []}
    again = [""] * len(numeric)
    for i, text in enumerate(data["text"]):
        if text.strip():
            middle = data["top"][i] + data["height"][i] / 2
            index = next((k for k, (y0, y1) in enumerate(spans) if y0 <= middle < y1), None)
            if index is not None:
                again[index] += text.strip()
    for word, second in zip(numeric, again):
        if re.sub(r"\D", "", word[0]) != re.sub(r"\D", "", second):
            word[1] = min(word[1], UNSURE_CONF)


def _text_layer_lines(page) -> list[_Line]:
    lines: dict[tuple[int, int], _Line] = {}
    for _x0, _y0, _x1, _y1, text, block, line, _no in page.get_text("words"):
        lines.setdefault((block, line), []).append((text, 100.0))
    return list(lines.values())


def _drop_page_furniture(pages: dict[int, list[_Line]]) -> int:
    """Убрать номера страниц и колонтитулы: строки у верхнего и нижнего края, которые
    похожи на номер страницы или повторяются (с точностью до цифр) на половине страниц."""
    def edge(lines: list[_Line]) -> list[int]:
        n = len(lines)
        return sorted({i for i in (0, 1, n - 2, n - 1) if 0 <= i < n})

    def text(line: _Line) -> str:
        return " ".join(w for w, _c in line).strip()

    def shape(line: _Line) -> str:
        return re.sub(r"\d+", "#", word_key(text(line).replace(" ", "")))

    seen = Counter()
    for lines in pages.values():
        seen.update({shape(lines[i]) for i in edge(lines)})
    repeated = {s for s, n in seen.items() if s and len(pages) >= 3 and n >= max(2, len(pages) / 2)}
    dropped = 0
    for page, lines in pages.items():
        drop = {i for i in edge(lines) if _PAGE_NUMBER_RE.match(text(lines[i])) or shape(lines[i]) in repeated}
        dropped += len(drop)
        pages[page] = [line for i, line in enumerate(lines) if i not in drop]
    return dropped


def load_pdf(content: bytes, filename: str) -> TextDoc:
    """Слова PDF в порядке чтения: текстовый слой, а для сканов — распознавание с уверенностью."""
    _check_size(content, MAX_FILE_SIZE_PDF)
    import pymupdf

    try:
        pdf = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise DocumentError(f"Не удалось прочитать «{filename}» как PDF: {exc}") from exc
    if pdf.needs_pass:
        raise DocumentError(f"«{filename}» защищён паролем — снимите защиту и загрузите снова")
    if pdf.page_count > MAX_PDF_PAGES:
        raise DocumentError(f"В «{filename}» слишком много страниц ({pdf.page_count}), лимит — {MAX_PDF_PAGES}.")
    pages: dict[int, list[_Line]] = {}
    scanned: list[int] = []
    for index in range(pdf.page_count):
        page = pdf[index]
        if len((page.get_text() or "").strip()) >= MIN_TEXT_LAYER_CHARS or not page.get_images():
            pages[index + 1] = _join_hyphens(_text_layer_lines(page))
            continue
        scanned.append(index + 1)
        if len(scanned) > MAX_SCANNED_PAGES:
            raise DocumentError(
                f"В «{filename}» больше {MAX_SCANNED_PAGES} страниц-сканов: распознавание займёт слишком долго. "
                "Сравните документ частями."
            )
        lines = _ocr_page(page)
        if lines is None:
            raise DocumentError(
                f"«{filename}» — скан, а распознавание (Tesseract) на сервере не установлено: сравнить нельзя."
            )
        pages[index + 1] = _join_hyphens(lines)
    doc = TextDoc(filename=filename, words=[], pages=pdf.page_count, scanned_pages=scanned)
    doc.dropped_lines = _drop_page_furniture(pages)
    for page_no, lines in pages.items():
        for line in lines:
            for text, conf in line:
                key = word_key(text)
                if key:
                    doc.words.append(Word(text, key, f"стр. {page_no}", conf, page_no in scanned))
    return _checked(doc)


def load_word(content: bytes, filename: str) -> TextDoc:
    """Слова Word в порядке документа: абзацы и таблицы на своих местах (колонтитулы не входят)."""
    _check_size(content, MAX_FILE_SIZE_WORD)
    lower = filename.lower()
    doc = TextDoc(filename=filename, words=[])
    if lower.endswith(".doc"):
        for n, line in enumerate((ln for ln in dp._doc_to_text(content).splitlines() if ln.strip()), 1):
            _add_words(doc, line, f"абз. {n}")
        return _checked(doc)
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = Document(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise DocumentError(f"Не удалось прочитать «{filename}» как .docx: {exc}") from exc
    paragraph_no = table_no = 0
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, document).text
            if text.strip():
                paragraph_no += 1
                _add_words(doc, text, f"абз. {paragraph_no}")
        elif tag == "tbl":
            table_no += 1
            for row in Table(child, document).rows:
                cells: list[str] = []
                for cell in row.cells:  # объединённые ячейки python-docx повторяет — берём один раз
                    if not cells or cell.text != cells[-1]:
                        cells.append(cell.text)
                _add_words(doc, " ".join(cells), f"табл. {table_no}")
    return _checked(doc)


def _add_words(doc: TextDoc, text: str, where: str) -> None:
    for token in text.split():
        key = word_key(token)
        if key:
            doc.words.append(Word(token, key, where))


def _checked(doc: TextDoc) -> TextDoc:
    if not doc.words:
        raise DocumentError(f"В «{doc.filename}» не найдено текста для сравнения.")
    if len(doc.words) > MAX_WORDS:
        raise DocumentError(f"«{doc.filename}» слишком большой для сравнения ({len(doc.words)} слов, лимит — {MAX_WORDS}).")
    return doc


def load_document(content: bytes, filename: str) -> TextDoc:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return load_pdf(content, filename)
    if lower.endswith((".docx", ".doc")):
        return load_word(content, filename)
    raise DocumentError(f"«{filename}»: сравнивать по тексту можно PDF и Word")


# --------------------------------- сравнение ---------------------------------


@dataclass
class Change:
    status: str
    old: list[Word]
    new: list[Word]
    before: list[Word]  # контекст до изменения
    after: list[Word]  # контекст после
    where_old: str
    where_new: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class TextComparison:
    old: TextDoc
    new: TextDoc
    changes: list[Change]
    same_document: bool
    shared: float  # доля трёхсловий меньшего документа, которые есть в другом
    order_note: str
    truncated: bool = False
    seconds: float = 0.0

    def count(self, status: str) -> int:
        return sum(1 for c in self.changes if c.status == status)


SHORT_TEXT = 20  # в тексте короче — сравниваем наборы слов, трёхсловий слишком мало


def shared_share(a: list[str], b: list[str]) -> float:
    """Доля трёхсловий меньшего документа, которые встречаются в другом (0…1)."""
    size = SHINGLE if min(len(a), len(b)) >= SHORT_TEXT else 1

    def grams(keys: list[str]) -> set[tuple[str, ...]]:
        return {tuple(keys[i:i + size]) for i in range(len(keys) - size + 1)}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


def _where(words: list[Word], start: int, end: int) -> str:
    """Где фрагмент в документе; для пустого (вставка) — место вставки."""
    if not words:
        return ""
    if start >= end:
        return words[min(start, len(words) - 1)].where
    first, last = words[start].where, words[end - 1].where
    return first if first == last else f"{first} – {last}"


SURE_FRAGMENT = 4  # добавленный/удалённый текст на скане — не короче стольких слов
SHORT_CHANGE = 3  # замена «слово на слово» на скане — не длиннее стольких слов


def _is_number(key: str) -> bool:
    return bool(re.fullmatch(r"[\d.]+", key))


def classify(old: list[Word], new: list[Word], scan: bool) -> tuple[str, list[str]]:
    """Статус фрагмента. Текстовый слой и Word точны — различие утверждается как есть.

    Если рядом скан (хотя бы в одной версии), утверждается только то, что на ошибку
    распознавания не похоже: короткая замена «по форме» (число на число, слово на слово)
    уверенно прочитанных слов с разным написанием либо связный фрагмент от SURE_FRAGMENT
    слов. Остальное — «проверить»: пропуск слова распознаванием выглядит так же, как
    удаление, а перепутанный порядок шапки — как замена."""
    status = STATUS_CHANGED if old and new else STATUS_ADDED if new else STATUS_REMOVED
    if not scan:
        return status, []
    reasons = []
    unsure = [w.text for w in old + new if w.scanned and w.conf < SURE_CONF]
    if unsure:
        reasons.append("на скане прочитано неуверенно: " + ", ".join(unsure[:5]))
    if old and new:
        pairs = list(zip(old, new))
        if len(old) != len(new) or len(old) > SHORT_CHANGE:
            reasons.append("на скане фрагменты разной длины — возможен пропуск или другой порядок распознавания")
        elif any(_is_number(a.key) != _is_number(b.key) for a, b in pairs):
            reasons.append("на скане число против слова — возможна ошибка распознавания")
        elif any(bool(re.search("[a-z]", a.key)) != bool(re.search("[a-z]", b.key)) for a, b in pairs):
            reasons.append("на скане латиница против кириллицы («LUT» — «шт») — возможна ошибка распознавания")
        elif any(not _is_number(a.key) and SequenceMatcher(None, a.key, b.key).ratio() >= LOOKS_LIKE_OCR
                 for a, b in pairs):
            reasons.append("написание почти совпадает — возможна ошибка распознавания")
    else:
        words = old or new
        if len(words) < SURE_FRAGMENT or sum(1 for w in words if not _is_number(w.key) and len(w.key) >= 3) < 2:
            reasons.append("короткий фрагмент рядом со сканом — возможен пропуск распознавания")
    return (STATUS_CHECK if reasons else status), reasons


def compare(first: TextDoc, second: TextDoc, *, force: bool = False) -> TextComparison:
    """Сравнить две версии. Непохожие документы не сравниваются (если не force)."""
    from app.services.spec_compare import order_files

    started = time.time()
    swap, order_note = order_files(first.filename, second.filename)
    old, new = (second, first) if swap else (first, second)
    a, b = [w.key for w in old.words], [w.key for w in new.words]
    shared = shared_share(a, b)
    same = shared >= SAME_DOC_SHARE
    result = TextComparison(old, new, [], same, shared, order_note)
    if not same and not force:
        return result
    if not same and len(a) * len(b) > FORCED_MAX_CELLS:
        raise DocumentError("Документы слишком разные и большие, чтобы сравнить их по словам.")

    hunks: list[list[int]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if hunks and i1 - hunks[-1][1] <= MERGE_GAP and j1 - hunks[-1][3] <= MERGE_GAP:
            hunks[-1][1], hunks[-1][3] = i2, j2
        else:
            hunks.append([i1, i2, j1, j2])
    if len(hunks) > MAX_DIFF_ITEMS:
        hunks, result.truncated = hunks[:MAX_DIFF_ITEMS], True
    for i1, i2, j1, j2 in hunks:
        old_words, new_words = old.words[i1:i2], new.words[j1:j2]
        nearby = old.words[max(i1 - 1, 0):i2 + 1] + new.words[max(j1 - 1, 0):j2 + 1]
        status, reasons = classify(old_words, new_words, scan=any(w.scanned for w in nearby))
        result.changes.append(Change(
            status, old_words, new_words,
            before=new.words[max(j1 - CONTEXT, 0):j1], after=new.words[j2:j2 + CONTEXT],
            where_old=_where(old.words, i1, i2), where_new=_where(new.words, j1, j2), reasons=reasons,
        ))
    result.seconds = round(time.time() - started, 1)
    return result


# ------------------------------- отчёт (Word) --------------------------------

HOW_TO_READ = [
    "Тексты сравнены по словам в порядке чтения; переносы строк, регистр букв, знаки препинания и номера "
    "страниц не учитываются.",
    "В колонке «Было» красным зачёркнутым — удалённое, в «Стало» зелёным — добавленное; серым — соседние "
    "слова для ориентира.",
    "«Проверить» — различие найдено на скане, но слово прочитано неуверенно или написание почти совпадает "
    "(«дотовор» — «договор»): вероятна ошибка распознавания. Сверьте со сканом.",
    "Колонтитулы и номера страниц, повторяющиеся на листах, в сравнение не входят. Форматирование "
    "(шрифт, выделение) не сравнивается.",
]


def _clean(text: str) -> str:
    """Убрать управляющие символы, которые ломают XML документа Word."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def build_compare_docx(result: TextComparison) -> bytes:
    from docx import Document
    from docx.shared import Pt, RGBColor

    grey, red, green = RGBColor(0x80, 0x80, 0x80), RGBColor(0xC0, 0x00, 0x00), RGBColor(0x00, 0x80, 0x00)
    doc = Document()
    doc.add_heading("Сравнение документов", level=1)
    for label, value in [
        ("Старая версия", f"{result.old.filename} ({_pages_note(result.old)})"),
        ("Новая версия", f"{result.new.filename} ({_pages_note(result.new)})"),
        ("Какая старая", result.order_note),
        ("Общий текст", f"{round(result.shared * 100)} % трёхсловий меньшего документа есть в другом"
                        + ("" if result.same_document else " — не похоже на версии одного документа")),
        ("Изменений", ", ".join(f"{s} — {result.count(s)}" for s in
                                (STATUS_CHANGED, STATUS_ADDED, STATUS_REMOVED, STATUS_CHECK))),
    ]:
        para = doc.add_paragraph()
        para.add_run(f"{label}: ").bold = True
        para.add_run(_clean(value))
    if not result.changes:
        doc.add_paragraph("Различий в тексте нет.")
    else:
        table = doc.add_table(rows=1, cols=5)
        table.style = "Table Grid"
        for cell, title in zip(table.rows[0].cells, ["№", "Где (стар. → нов.)", "Было", "Стало", "Статус"]):
            cell.paragraphs[0].add_run(title).bold = True
        for n, change in enumerate(result.changes, 1):
            cells = table.add_row().cells
            cells[0].text = str(n)
            cells[1].text = _clean(f"{change.where_old} → {change.where_new}")
            for cell, words, color, strike in ((cells[2], change.old, red, True), (cells[3], change.new, green, False)):
                para = cell.paragraphs[0]
                _run(para, " ".join(w.text for w in change.before[-3:]), grey)
                run = _run(para, " ".join(w.text for w in words) or "—", color)
                run.font.strike = strike and bool(words)
                run.bold = not strike and bool(words)
                _run(para, " ".join(w.text for w in change.after[:3]), grey)
            cells[4].text = change.status
            if change.reasons:
                note = cells[4].add_paragraph().add_run(_clean("; ".join(change.reasons)))
                note.italic, note.font.size = True, Pt(8)
    doc.add_heading("Как читать", level=2)
    for line in HOW_TO_READ:
        doc.add_paragraph(line, style="List Bullet")
    if result.truncated:
        doc.add_paragraph(f"Показаны первые {MAX_DIFF_ITEMS} изменений.")
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _run(para, text: str, color):
    run = para.add_run(_clean(text) + " " if text else "")
    run.font.color.rgb = color
    return run


def _pages_note(doc: TextDoc) -> str:
    if not doc.pages:
        return f"{len(doc.words)} слов"
    note = f"{doc.pages} стр."
    if doc.scanned_pages:
        note += f", сканов распознано: {len(doc.scanned_pages)}"
    return note
