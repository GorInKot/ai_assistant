"""Сравнение двух редакций спецификации (форма ГОСТ 21.110) по позициям.

Обе редакции разбираются реестром позиций (`spec_registry`), затем позиции сопоставляются
с сохранением порядка: по номеру позиции, если он есть, и по наименованию — с допуском
на ошибки распознавания. У сопоставленных позиций сравниваются единица, количество и масса.

Как и в реестре, ничего не додумывается: разница, где хоть одно из значений прочитано
неуверенно (правка от руки, штамп, плохой скан), — не «изменено», а «проверить».

Сначала проверяется, что это вообще версии одного документа (шифр по штампу, доля
совпавших позиций и совпавших количеств). Листы, которым в другой версии нет пары
(изменение часто выпускают только на заменяемые листы), выносятся отдельно, а не
считаются добавленными или удалёнными позициями.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher

import numpy as np

from app.services import spec_registry as sr
from app.services.spec_registry import COL_MASS, COL_NAME, COL_POS, COL_QTY, COL_TYPE, COL_UNIT, Position, SpecDocument

MATCH_MIN = 0.6  # сходство позиций ниже — это разные позиции
MOVED_MIN = 0.85  # позиция, переставленная в другое место, сопоставляется, только если почти совпадает
PREFILTER_MIN = 0.2  # доля общих триграмм наименований, ниже — сходство не считаем (ускорение)
SAME_DOC_MATCHED = 0.5  # сопоставлено не меньше половины позиций меньшей версии…
SAME_DOC_EQUAL_QTY = 0.3  # …и у трети сопоставленных совпадает количество — версии одного документа

STATUS_SAME = "без изменений"
STATUS_CHANGED = "изменено"
STATUS_CHECK = "проверить"
STATUS_ADDED = "добавлено"
STATUS_REMOVED = "удалено"
STATUS_NEW_SHEET = "лист только в новой"
STATUS_OLD_SHEET = "лист только в старой"

_LOOKALIKES = str.maketrans("abcehkmoptxy", "авсенкмортху")


def _key(text: str) -> str:
    """Текст для сравнения: без регистра, пробелов и знаков, латинские двойники → кириллица."""
    text = (text or "").lower().replace("ё", "е").translate(_LOOKALIKES)
    return re.sub(r"[^0-9а-яa-z]", "", text)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _trigrams(text: str) -> set[str]:
    return {text[i:i + 3] for i in range(max(len(text) - 2, 1))} if text else set()


@dataclass
class _Item:
    """Позиция с заранее посчитанными ключами для сопоставления."""

    position: Position
    name: str
    type_: str
    pos: str
    grams: set[str]
    qty: float | None

    @classmethod
    def of(cls, position: Position) -> "_Item":
        name = _key(position.cells[COL_NAME])
        return cls(position, name, _key(position.cells[COL_TYPE]), _key(position.cells[COL_POS]), _trigrams(name),
                   _number(position.cells[COL_QTY]))


def similarity(a: _Item, b: _Item) -> float:
    """Сходство двух позиций 0…~1,6: наименование, марка, номер позиции и количество.

    Наименования в двух сканах читаются с разными ошибками (а вариант бывает прочитан то
    полностью, то одним размером), поэтому совпавшие номер позиции и количество вместе
    достаточны для пары и без похожего наименования."""
    same_pos = bool(a.pos and a.pos == b.pos)
    same_qty = a.qty is not None and a.qty == b.qty
    if not same_pos:
        union = len(a.grams | b.grams)
        if not union or len(a.grams & b.grams) / union < PREFILTER_MIN:
            return 0.0
    score = _ratio(a.name, b.name)
    short, longer = sorted((a.name, b.name), key=len)
    if len(short) >= 3 and longer.endswith(short):  # «4х2,5» против полного наименования того же варианта
        score = max(score, 0.7)
    if a.type_ and b.type_:
        score = 0.8 * score + 0.2 * _ratio(a.type_, b.type_)
    if a.pos and b.pos:
        score += 0.3 if same_pos else -0.15
    if same_qty:
        score += 0.3
    elif (a.qty is None) != (b.qty is None):
        score -= 0.1  # строка без количества (заголовок группы) против позиции с количеством
    return score


def align(old: list[_Item], new: list[_Item]) -> tuple[list[tuple[int | None, int | None]], set[int]]:
    """Сопоставить позиции с сохранением порядка: максимум суммарного сходства.

    Возвращает пары индексов (старая, новая) — None, если позиции нет пары, — и индексы
    старых позиций, сопоставленных вне порядка (переставлены в документе)."""
    n, m = len(old), len(new)
    sim = np.zeros((n, m))
    for i, a in enumerate(old):
        for j, b in enumerate(new):
            sim[i, j] = similarity(a, b)
    best = np.zeros((n + 1, m + 1))
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            value = max(best[i + 1, j], best[i, j + 1])
            if sim[i, j] >= MATCH_MIN:
                value = max(value, best[i + 1, j + 1] + sim[i, j])
            best[i, j] = value
    pairs: list[tuple[int | None, int | None]] = []
    i = j = 0
    while i < n and j < m:
        if sim[i, j] >= MATCH_MIN and best[i, j] == best[i + 1, j + 1] + sim[i, j]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i, j] == best[i + 1, j]:
            pairs.append((i, None))
            i += 1
        else:
            pairs.append((None, j))
            j += 1
    pairs += [(k, None) for k in range(i, n)] + [(None, k) for k in range(j, m)]
    return _match_moved(pairs, sim)


def _match_moved(pairs, sim) -> tuple[list[tuple[int | None, int | None]], set[int]]:
    """Позиции, переставленные в другое место: вне порядка — только почти совпадающие."""
    lonely_old = [i for i, j in pairs if j is None]
    lonely_new = [j for i, j in pairs if i is None]
    candidates = sorted(((sim[i, j], i, j) for i in lonely_old for j in lonely_new if sim[i, j] >= MOVED_MIN),
                        reverse=True)
    moved: dict[int, int] = {}
    used: set[int] = set()
    for _score, i, j in candidates:
        if i not in moved and j not in used:
            moved[i] = j
            used.add(j)
    if not moved:
        return pairs, set()
    by_new = {j: i for i, j in moved.items()}
    result = []
    for i, j in pairs:
        if j is None and i in moved:
            continue  # стоит теперь там, где её новое место
        result.append((by_new[j], j) if i is None and j in by_new else (i, j))
    return result, set(moved)


def _number(text: str) -> float | None:
    value = (text or "").strip().rstrip("*").replace(",", ".")
    return float(value) if re.fullmatch(r"\d+(?:\.\d+)?", value) else None


def _same(col: int, a: str, b: str) -> bool:
    if col == COL_UNIT:
        return a.strip().rstrip(".").lower() == b.strip().rstrip(".").lower()
    na, nb = _number(a), _number(b)
    if na is not None and nb is not None:
        return abs(na - nb) < 1e-9
    return _key(a) == _key(b)


_FIELDS = ((COL_UNIT, "ед."), (COL_QTY, "кол-во"), (COL_MASS, "масса"))
_DIMENSION_RE = re.compile(r"\d+(?:[.,]\d+)?(?:х\d+(?:[.,]\d+)?)+")


def dimensions(name: str) -> list[str]:
    """Размеры вида «325х8,0», «4х2,5» из наименования (пробелы вокруг знака «х» убраны)."""
    text = re.sub(r"(\d)\s*[xх×]\s*(\d)", r"\1х\2", name.lower())
    return sorted(t.replace(",", ".") for t in _DIMENSION_RE.findall(text))


def differences(old: Position, new: Position) -> tuple[list[str], list[str]]:
    """Что изменилось: единица, количество, масса — и отличия размеров в наименовании.

    Размеры сравниваются, только если прочитаны в обеих версиях: прочие числа в
    наименовании (номера ГОСТ, марки) в двух сканах читаются по-разному слишком часто."""
    values = [
        f"{label}: {old.cells[col] or '—'} → {new.cells[col] or '—'}"
        for col, label in _FIELDS if not _same(col, old.cells[col], new.cells[col])
    ]
    was, now = dimensions(old.cells[COL_NAME]), dimensions(new.cells[COL_NAME])
    names = []
    if was and now and was != now:
        gone = [t for t in was if t not in now]
        came = [t for t in now if t not in was]
        names.append(f"размер в наименовании: {' '.join(gone) or '—'} → {' '.join(came) or '—'}")
    return values, names


@dataclass
class Row:
    status: str
    old: Position | None = None
    new: Position | None = None
    changes: list[str] = field(default_factory=list)  # «кол-во: 129 → 100»
    reasons: list[str] = field(default_factory=list)  # почему проверить вручную
    moved: bool = False


@dataclass
class SpecComparison:
    old: SpecDocument
    new: SpecDocument
    rows: list[Row]
    same_document: bool
    evidence: str  # почему считаем (или нет) версиями одного документа
    order_note: str  # как определили, какая версия старая
    excluded: int = 0  # позиций, зачёркнутых целиком, — в сравнении не участвуют
    seconds: float = 0.0

    def count(self, status: str) -> int:
        return sum(1 for r in self.rows if r.status == status)

    def pages(self, status: str) -> list[int]:
        side = "new" if status == STATUS_NEW_SHEET else "old"
        return sorted({getattr(r, side).page for r in self.rows if r.status == status})


def _row(old: Position, new: Position, moved: bool) -> Row:
    values, names = differences(old, new)
    if not values and not names:
        return Row(STATUS_SAME, old, new, moved=moved)
    reasons = [f"стар.: {r}" for r in old.issues] + [f"нов.: {r}" for r in new.issues]
    if names:  # другой размер — возможно, другая позиция или ошибка чтения: не утверждаем
        reasons.append("размер в наименовании прочитан по-разному — сверьте со сканами")
    status = STATUS_CHECK if reasons else STATUS_CHANGED
    return Row(status, old, new, values + names, reasons, moved)


def _code_key(code: str) -> str:
    key = re.sub(r"[^0-9A-ZА-ЯЁ]", "", code.upper()).translate(sr._CODE_LOOKALIKES)
    return key.replace("О", "0").replace("З", "3")


def _code_read(doc: SpecDocument) -> bool:
    return bool(doc.code) and doc.code != doc.filename.rsplit(".", 1)[0]


def same_document(old: SpecDocument, new: SpecDocument, rows: list[Row]) -> tuple[bool, str]:
    """Версии одного документа? Шифр по штампу совпал — да. Иначе по содержимому:
    сопоставлена большая часть позиций меньшей версии и у многих совпало количество."""
    matched = [r for r in rows if r.old is not None and r.new is not None]
    smaller = min(sum(1 for r in rows if r.old is not None), sum(1 for r in rows if r.new is not None))
    share = len(matched) / smaller if smaller else 0.0
    equal_qty = sum(1 for r in matched if _same(COL_QTY, r.old.cells[COL_QTY], r.new.cells[COL_QTY]))
    qty_share = equal_qty / len(matched) if matched else 0.0
    facts = f"сопоставлено {len(matched)} из {smaller} позиций, количество совпало у {equal_qty}"
    if _code_read(old) and _code_read(new) and _code_key(old.code) == _code_key(new.code):
        return True, f"шифр по штампу совпадает ({new.code}); {facts}"
    if share >= SAME_DOC_MATCHED and qty_share >= SAME_DOC_EQUAL_QTY:
        return True, f"по содержимому: {facts}"
    codes = f"шифры по штампу: {old.code} и {new.code}; " if _code_read(old) and _code_read(new) else ""
    return False, codes + facts


_REVISION_RES = (
    ("R", re.compile(r"(?:^|[_\s\-])R(\d{1,3})(?=[_\s\-.]|$)")),
    ("rC", re.compile(r"(?:^|[_\s\-])rC(\d{1,3})(?=[_\s\-.]|$)", re.IGNORECASE)),
    ("изм.", re.compile(r"изм\.?\s*(\d{1,3})", re.IGNORECASE)),
    ("ред.", re.compile(r"ред(?:акция)?\.?\s*(\d{1,3})", re.IGNORECASE)),
)


def revision(filename: str) -> tuple[str, int] | None:
    """Номер редакции из имени файла: «…_R04.pdf» → ("R", 4), «… изм.3.pdf» → ("изм.", 3)."""
    for scheme, pattern in _REVISION_RES:
        found = pattern.search(filename)
        if found:
            return scheme, int(found.group(1))
    return None


ORDER_BY_USER = "старую версию указали вы"
ORDER_BY_ATTACHMENT = "старой считается первый приложенный файл: по именам файлов порядок не определить"

# Слова в имени файла, которые прямо говорят, какая версия старая, а какая новая.
_OLD_WORDS = {"старый", "старая", "старое", "стар", "прежний", "прежняя", "прежнее", "исходный", "исходная",
              "исходное", "было", "old", "orig", "original"}
_NEW_WORDS = {"новый", "новая", "новое", "нов", "исправленный", "исправленная", "исправленное", "стало",
              "new", "final", "финал"}
_DATE_RES = (
    re.compile(r"(?<!\d)(\d{2})[._-](\d{2})[._-](\d{4})(?!\d)"),  # 15.10.2026
    re.compile(r"(?<!\d)(\d{4})[._-](\d{2})[._-](\d{2})(?!\d)"),  # 2026-10-15
)


def _name_words(filename: str) -> set[str]:
    stem = filename.rsplit(".", 1)[0].lower()
    return set(re.split(r"[^a-zа-яё0-9]+", stem)) - {""}


def _name_date(filename: str) -> tuple[int, int, int] | None:
    """Единственная корректная дата в имени файла (год, месяц, день), иначе None."""
    found = set()
    for i, pattern in enumerate(_DATE_RES):
        for m in pattern.finditer(filename):
            d, mo, y = (m.group(1), m.group(2), m.group(3)) if i == 0 else (m.group(3), m.group(2), m.group(1))
            try:
                day = date(int(y), int(mo), int(d))  # 31.02 — не дата
            except ValueError:
                continue
            if 1990 <= day.year <= 2100:
                found.add((day.year, day.month, day.day))
    return found.pop() if len(found) == 1 else None


def order_by_names(first: str, second: str) -> tuple[bool, str] | None:
    """Какой файл старый — по однозначным признакам в именах: номер редакции (одна схема
    нумерации), слова «старый / новый», даты. (поменять ли местами, пояснение) или None."""
    a, b = revision(first), revision(second)
    if a and b and a[0] == b[0] and a[1] != b[1]:
        return a[1] > b[1], "определено по номеру редакции в именах файлов"
    wa, wb = _name_words(first), _name_words(second)
    first_old = bool(wa & _OLD_WORDS or wb & _NEW_WORDS)
    second_old = bool(wb & _OLD_WORDS or wa & _NEW_WORDS)
    if first_old != second_old:
        return second_old, "определено по словам «старый / новый» в именах файлов"
    da, db = _name_date(first), _name_date(second)
    if da and db and da != db:
        return da > db, "определено по датам в именах файлов"
    return None


def order_files(first: str, second: str, *, chosen: bool = False) -> tuple[bool, str]:
    """Какой файл старый. chosen — пользователь сам указал, что старый — первый. Иначе по
    именам файлов, а если по ним не определить — первый приложенный (в чате до этого не
    доходит: ассистент спрашивает). Возвращает (поменять ли местами, пояснение)."""
    if chosen:
        return False, ORDER_BY_USER
    return order_by_names(first, second) or (False, ORDER_BY_ATTACHMENT)


def order_documents(
    first: SpecDocument, second: SpecDocument, *, chosen: bool = False
) -> tuple[SpecDocument, SpecDocument, str]:
    swap, note = order_files(first.filename, second.filename, chosen=chosen)
    return (second, first, note) if swap else (first, second, note)


def compare(first: SpecDocument, second: SpecDocument, *, chosen: bool = False) -> SpecComparison:
    """Сравнить две разобранные редакции спецификации. chosen — старую указал пользователь (первая)."""
    started = time.time()
    old, new, order_note = order_documents(first, second, chosen=chosen)
    old_items = [_Item.of(p) for p in old.positions if not p.excluded]
    new_items = [_Item.of(p) for p in new.positions if not p.excluded]
    rows: list[Row] = []
    pairs, moved = align(old_items, new_items)
    for i, j in pairs:
        if i is not None and j is not None:
            rows.append(_row(old_items[i].position, new_items[j].position, moved=i in moved))
        elif i is not None:
            rows.append(Row(STATUS_REMOVED, old=old_items[i].position))
        else:
            rows.append(Row(STATUS_ADDED, new=new_items[j].position))
    _mark_unpaired_sheets(rows)
    _mark_unsure_lonely(rows)
    same, evidence = same_document(old, new, rows)
    excluded = old.excluded + new.excluded
    return SpecComparison(old, new, rows, same, evidence, order_note, excluded, round(time.time() - started, 1))


PAIRED_SHEET_MIN = 2  # столько сопоставленных позиций (и не меньше пятой части) — лист есть в обеих версиях


def _paired_pages(rows: list[Row], side: str) -> set[int]:
    from collections import Counter

    total = Counter(getattr(r, side).page for r in rows if getattr(r, side) is not None)
    matched = Counter(getattr(r, side).page for r in rows if r.old is not None and r.new is not None)
    return {page for page, n in matched.items() if n >= min(PAIRED_SHEET_MIN, total[page]) and n >= 0.2 * total[page]}


def _mark_unpaired_sheets(rows: list[Row]) -> None:
    """Позиции с листов, у которых в другой версии нет пары, — не «добавлены/удалены»,
    а «лист только в старой/новой»: изменение могли выпустить только на заменяемые листы.
    Одно случайное совпадение лист парным не делает."""
    paired_old, paired_new = _paired_pages(rows, "old"), _paired_pages(rows, "new")
    for r in rows:
        if r.status == STATUS_REMOVED and r.old.page not in paired_old:
            r.status = STATUS_OLD_SHEET
        elif r.status == STATUS_ADDED and r.new.page not in paired_new:
            r.status = STATUS_NEW_SHEET


def _mark_unsure_lonely(rows: list[Row]) -> None:
    """Позиция без пары, прочитанная неуверенно (обрывок правки от руки, штамп, строка,
    разорванная распознаванием), — не «добавлено/удалено», а «проверить»."""
    for r in rows:
        item = r.old if r.status == STATUS_REMOVED else r.new if r.status == STATUS_ADDED else None
        if item is not None and item.issues:
            side = "старой" if r.status == STATUS_REMOVED else "новой"
            r.changes = [f"есть только в {side} редакции"]
            r.reasons = ["нет пары в другой редакции, а позиция прочитана неуверенно"] + item.issues
            r.status = STATUS_CHECK


# ---------------------------------- Excel -----------------------------------

EXCEL_HEADERS = [
    "Статус", "Стр. стар.", "Стр. нов.", "Раздел", "Позиция", "Наименование и техническая характеристика",
    "Тип, марка", "Ед. стар.", "Ед. нов.", "Кол-во стар.", "Кол-во нов.", "Разница", "Масса стар.",
    "Масса нов.", "Что изменилось", "Проверить",
]
EXCEL_WIDTHS = [16, 7, 7, 22, 9, 60, 26, 7, 7, 10, 10, 9, 9, 9, 34, 40]

FILLS = {
    STATUS_CHANGED: "FCE4D6",
    STATUS_CHECK: "FFF2B3",
    STATUS_ADDED: "E2EFDA",
    STATUS_NEW_SHEET: "EDEDED",
    STATUS_OLD_SHEET: "EDEDED",
}

HOW_TO_READ = [
    "Сравнение собрано по сканам двух редакций спецификации (форма ГОСТ 21.110) без участия языковой модели.",
    "Позиции сопоставлены по номеру позиции и наименованию с допуском на ошибки распознавания, порядок сохранён.",
    "Оранжевым — «изменено»: единица, количество или масса различаются, и в обеих редакциях значения прочитаны уверенно.",
    "Жёлтым — «проверить»: значения различаются, но хотя бы в одной редакции прочитаны неуверенно (правка от руки, "
    "штамп, плохой скан) или различаются только числа в наименовании. Сверьте со сканами.",
    "Зелёным — «добавлено» (позиции нет в старой редакции); серым зачёркнутым — «удалено» (нет в новой).",
    "Серым — «лист только в старой/новой»: у листа нет ни одной общей позиции с другой редакцией. Так бывает, "
    "когда изменение выпущено только на заменяемые листы, — это не обязательно добавленные или удалённые позиции.",
    "«Разница» = количество в новой редакции минус в старой, если оба значения — числа.",
    "Позиции, зачёркнутые целиком (исключены изменением), в сравнении не участвуют.",
    "Шифр прочитан из основной надписи распознаванием и может содержать ошибки.",
]


def _difference(row: Row):
    if row.old is None or row.new is None:
        return ""
    a, b = _number(row.old.cells[COL_QTY]), _number(row.new.cells[COL_QTY])
    return round(b - a, 6) if a is not None and b is not None and a != b else ""


def build_compare_xlsx(result: SpecComparison) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    from app.services.document_processing import append_row

    wb = Workbook()
    ws = wb.active
    ws.title = "Сравнение"
    append_row(ws, EXCEL_HEADERS)
    head_fill = PatternFill("solid", fgColor="E7EEF8")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = head_fill
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    removed_font = Font(color="808080", strike=True)
    for row in result.rows:
        main = row.new or row.old
        old, new = row.old.cells if row.old else [""] * 9, row.new.cells if row.new else [""] * 9
        status = row.status + (" (переставлена)" if row.moved else "")
        append_row(ws, [
            status, row.old.page if row.old else "", row.new.page if row.new else "", main.section,
            main.cells[COL_POS], main.cells[COL_NAME], main.cells[COL_TYPE], old[COL_UNIT], new[COL_UNIT],
            sr._as_number(old[COL_QTY]), sr._as_number(new[COL_QTY]), _difference(row),
            sr._as_number(old[COL_MASS]), sr._as_number(new[COL_MASS]), "; ".join(row.changes), "; ".join(row.reasons),
        ])
        fill = FILLS.get(row.status)
        for cell in ws[ws.max_row]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if fill:
                cell.fill = PatternFill("solid", fgColor=fill)
            if row.status in (STATUS_REMOVED, STATUS_OLD_SHEET):
                cell.font = removed_font
    for i, width in enumerate(EXCEL_WIDTHS, 1):
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    summary = wb.create_sheet("Сводка")
    for label, value in [
        ("Старая редакция", result.old.filename),
        ("Шифр (по штампу)", result.old.code),
        ("Страниц с таблицей", result.old.table_pages),
        ("Новая редакция", result.new.filename),
        ("Шифр (по штампу)", result.new.code),
        ("Страниц с таблицей", result.new.table_pages),
        ("Какая старая", result.order_note),
        ("Версии одного документа", ("да — " if result.same_document else "не похоже — ") + result.evidence),
        *[(status, result.count(status)) for status in (
            STATUS_CHANGED, STATUS_CHECK, STATUS_ADDED, STATUS_REMOVED, STATUS_SAME, STATUS_NEW_SHEET, STATUS_OLD_SHEET)],
        ("Листы только в новой", ", ".join(map(str, result.pages(STATUS_NEW_SHEET)))),
        ("Листы только в старой", ", ".join(map(str, result.pages(STATUS_OLD_SHEET)))),
        ("Исключены изменением (не сравнивались)", result.excluded),
    ]:
        append_row(summary, [label, value])
    summary.column_dimensions["A"].width = 40
    summary.column_dimensions["B"].width = 90

    note = wb.create_sheet("Как читать")
    for line in HOW_TO_READ:
        append_row(note, [line])
    note.column_dimensions["A"].width = 120

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
