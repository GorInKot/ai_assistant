"""Обработка пользовательских документов (Этап 6 роадмапа).

Excel (`.xlsx`): извлечение данных (6.A.1), объединение нескольких файлов с
проверкой идентичности структуры (6.A.2), поиск различий между двумя файлами
(6.A.3).

Принципы:
- Stateless: файлы обрабатываются в памяти, на сервере не сохраняются.
  Это согласуется с локальным/безопасным характером продукта — корпоративные
  данные не оседают на диске.
- Без тяжёлых зависимостей: используем уже имеющийся `openpyxl`, без `pandas`.
- Лимиты на размер/строки/колонки, чтобы один большой файл не положил процесс.

Старый бинарный `.xls` вне скоупа первой итерации (openpyxl его не читает) —
для него отдаётся понятная ошибка с просьбой пересохранить в `.xlsx`.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field


# ---- Лимиты (значения по умолчанию; при нужде вынести в config) ----

# Лимиты размера разнесены по типам: openpyxl разворачивает .xlsx в объекты в
# памяти с коэффициентом ~10-40× от размера файла (узкое место RAM на одном
# процессе), тогда как текст .docx/.doc разворачивается скромнее. Поэтому Excel
# держим строже Word.
MAX_FILE_SIZE_XLSX = 15 * 1024 * 1024  # 15 МБ на Excel-файл
MAX_FILE_SIZE_WORD = 30 * 1024 * 1024  # 30 МБ на Word-файл (.docx/.doc)
MAX_ROWS = 50_000  # строк данных на лист (читаем не больше)
MAX_COLS = 256  # колонок на лист
MAX_FILES = 20  # файлов за одну операцию объединения/сравнения
PREVIEW_ROWS = 100  # строк в превью результата (полные данные — в скачиваемом файле)
MAX_DIFF_ITEMS = 1000  # ограничение размера отчёта о различиях
MAX_SUMMARY_CHARS = 30_000  # сколько символов текста отдаём в LLM на выжимку
DOC_CONVERT_TIMEOUT = 30  # сек на конвертацию старого .doc внешней утилитой
GRID_SCAN_ROWS = 1000  # сколько строк читаем «сеткой» при детекте формы (формы малы)
FORM_DENSITY_THRESHOLD = 0.30  # доля заполненных ячеек ниже которой лист считаем формой


class DocumentError(ValueError):
    """Ошибка обработки документа, безопасная для показа пользователю."""


def _check_size(content: bytes, limit: int) -> None:
    """Проверить размер файла против лимита (общий текст ошибки для xlsx/word)."""
    if len(content) > limit:
        raise DocumentError(
            f"Файл слишком большой ({len(content) // 1024} КБ). "
            f"Лимит — {limit // (1024 * 1024)} МБ."
        )


# ----------------------------- Модели данных -----------------------------


@dataclass
class SheetData:
    name: str
    headers: list[str]
    rows: list[list[str]]  # строки данных (без заголовка), уже обрезанные по лимиту
    total_rows: int  # всего строк данных в листе (до обрезки)
    truncated: bool  # True, если total_rows > len(rows)

    @property
    def n_cols(self) -> int:
        return len(self.headers)


@dataclass
class WorkbookData:
    filename: str
    sheets: list[SheetData]


@dataclass
class MergeResult:
    ok: bool
    headers: list[str]
    rows: list[list[str]]  # объединённые строки (полные)
    total_rows: int
    source_counts: list[tuple[str, int]]  # (filename, добавлено строк)
    mismatches: list[str] = field(default_factory=list)


@dataclass
class DiffResult:
    structure_changed: bool
    added_columns: list[str]
    removed_columns: list[str]
    reordered: bool
    key_column: str | None  # по какой колонке матчили строки (None → позиционно)
    added_rows: list[list[str]]
    removed_rows: list[list[str]]
    changed_cells: list[dict]  # {"key", "column", "old", "new"}
    truncated: bool  # отчёт обрезан по MAX_DIFF_ITEMS


# ------------------------------- Парсинг ---------------------------------


def _cell_to_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def parse_xlsx(
    content: bytes,
    filename: str,
    *,
    max_rows: int = MAX_ROWS,
    max_cols: int = MAX_COLS,
) -> WorkbookData:
    """Разобрать .xlsx в структурированное представление (листы/заголовки/строки)."""
    lower = (filename or "").lower()
    if lower.endswith(".xls") and not lower.endswith(".xlsx"):
        raise DocumentError(
            "Старый формат .xls не поддерживается. "
            "Пересохраните файл как .xlsx и загрузите снова."
        )
    if not lower.endswith(".xlsx"):
        raise DocumentError("Ожидается файл Excel в формате .xlsx")
    _check_size(content, MAX_FILE_SIZE_XLSX)

    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover
        raise DocumentError("openpyxl не установлен на сервере") from exc

    try:
        workbook = load_workbook(io.BytesIO(content), data_only=True, read_only=True)
    except Exception as exc:  # noqa: BLE001 — openpyxl кидает разнородные ошибки
        raise DocumentError(f"Не удалось прочитать файл как .xlsx: {exc}") from exc

    sheets: list[SheetData] = []
    for ws in workbook.worksheets:
        # Материализуем строки (с запасом по лимиту), чтобы корректно определить
        # ширину таблицы и строку заголовков — частый случай: первая строка это
        # объединённый заголовок-«шапка» (напр. «Сотрудники»), а реальные имена
        # колонок идут ниже.
        raw_rows: list[list[str]] = []
        for raw in ws.iter_rows(values_only=True):
            raw_rows.append([_cell_to_str(v) for v in raw])
            if len(raw_rows) >= max_rows + 1:
                break

        # Истинная ширина = максимум по «последней непустой» ячейке среди строк.
        width = min(max((_last_nonempty_len(r) for r in raw_rows), default=0), max_cols)
        if width == 0:
            sheets.append(SheetData(name=ws.title, headers=[], rows=[], total_rows=0, truncated=False))
            continue

        header_idx = _detect_header_index(raw_rows, width)
        if header_idx is None:
            sheets.append(SheetData(name=ws.title, headers=[], rows=[], total_rows=0, truncated=False))
            continue

        headers = _fit_row(raw_rows[header_idx], width)
        # Пустым заголовкам даём подпись, чтобы колонка с данными не «исчезала».
        headers = [h if h else f"Колонка {i + 1}" for i, h in enumerate(headers)]

        data_rows: list[list[str]] = []
        total = 0
        for raw in raw_rows[header_idx + 1:]:
            cells = _fit_row(raw, width)
            if all(c == "" for c in cells):
                continue
            total += 1
            if len(data_rows) < max_rows:
                data_rows.append(cells)

        sheets.append(
            SheetData(
                name=ws.title,
                headers=headers,
                rows=data_rows,
                total_rows=total,
                truncated=total > len(data_rows),
            )
        )

    workbook.close()
    return WorkbookData(filename=filename, sheets=sheets)


def _last_nonempty_len(cells: list[str]) -> int:
    end = len(cells)
    while end > 0 and cells[end - 1] == "":
        end -= 1
    return end


def _fit_row(cells: list[str], width: int) -> list[str]:
    row = list(cells[:width])
    if len(row) < width:
        row += [""] * (width - len(row))
    return row


def _detect_header_index(rows: list[list[str]], width: int) -> int | None:
    """Найти строку заголовков, пропустив «титульные» строки сверху.

    Заголовок — первая строка, заполненная достаточно плотно для текущей ширины
    (для таблиц шире одной колонки — минимум 2 непустые ячейки; это отсекает
    одноячеечные титулы вроде объединённой «шапки»). Для одноколоночной таблицы
    берём первую непустую строку.
    """
    threshold = 1 if width == 1 else 2
    for idx, row in enumerate(rows):
        non_empty = sum(1 for c in row[:width] if c != "")
        if non_empty >= threshold:
            return idx
    return None


# --------------------------- Сравнение структуры -------------------------


def _norm_header(value: str) -> str:
    """Нормализация заголовка для сравнения: регистр и кратные пробелы игнорируются."""
    return " ".join(value.strip().lower().split())


def _structure_signature(headers: list[str]) -> list[str]:
    return [_norm_header(h) for h in headers]


# ----------------------------- 6.A.2 Объединение -------------------------


def merge_workbooks(workbooks: list[WorkbookData]) -> MergeResult:
    """Слить первые листы нескольких файлов в один.

    Предусловие: совпадение *набора* колонок по нормализованным именам (регистр
    и кратные пробелы игнорируются). **Порядок колонок может отличаться** — строки
    каждого файла автоматически переставляются к порядку первого файла (он задаёт
    итоговую структуру). При расхождении набора имён объединение не выполняется —
    возвращается, в каком файле каких колонок не хватает или какие лишние.
    """
    if len(workbooks) < 2:
        raise DocumentError("Для объединения нужно минимум два файла")

    base = workbooks[0].sheets[0] if workbooks[0].sheets else None
    if base is None or not base.headers:
        raise DocumentError(f"В файле «{workbooks[0].filename}» нет данных на первом листе")

    base_sig = _structure_signature(base.headers)
    base_set = set(base_sig)
    mismatches: list[str] = []
    # План переупорядочивания: для каждого файла — индексы его колонок в порядке base.
    plans: list[tuple[str, SheetData, list[int]]] = []

    for wb in workbooks:
        sheet = wb.sheets[0] if wb.sheets else None
        if sheet is None or not sheet.headers:
            mismatches.append(f"«{wb.filename}»: нет данных на первом листе")
            continue
        sig = _structure_signature(sheet.headers)
        if len(set(sig)) != len(sig):
            mismatches.append(
                f"«{wb.filename}»: повторяющиеся имена колонок — переупорядочивание неоднозначно"
            )
            continue
        sheet_set = set(sig)
        if sheet_set != base_set:
            missing = [h for h, s in zip(base.headers, base_sig) if s not in sheet_set]
            extra = [h for h, s in zip(sheet.headers, sig) if s not in base_set]
            parts = []
            if missing:
                parts.append(f"не хватает колонок: {', '.join(missing)}")
            if extra:
                parts.append(f"лишние колонки: {', '.join(extra)}")
            mismatches.append(f"«{wb.filename}»: {'; '.join(parts)}")
            continue
        pos = {s: i for i, s in enumerate(sig)}
        plans.append((wb.filename, sheet, [pos[s] for s in base_sig]))

    if mismatches:
        return MergeResult(
            ok=False,
            headers=base.headers,
            rows=[],
            total_rows=0,
            source_counts=[],
            mismatches=mismatches,
        )

    merged_rows: list[list[str]] = []
    source_counts: list[tuple[str, int]] = []
    for filename, sheet, order in plans:
        merged_rows.extend([row[i] for i in order] for row in sheet.rows)
        source_counts.append((filename, len(sheet.rows)))

    return MergeResult(
        ok=True,
        headers=base.headers,
        rows=merged_rows,
        total_rows=len(merged_rows),
        source_counts=source_counts,
    )


# Строки, начинающиеся с этих символов, Excel при вводе вручную воспринимает как формулу.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def neutralize_cell(cell) -> None:
    """Сделать значение ячейки обычным текстом, а не формулой (защита от formula injection).

    Текст, попадающий в Excel из внешних документов (PDF, чужие таблицы, имена файлов),
    недоверенный: openpyxl превращает строку, начинающуюся с «=», в живую формулу
    (`=HYPERLINK(...)`, `=WEBSERVICE(...)`, DDE), которая сработает у того, кто откроет файл.
    Фиксируем тип «строка»; для значений с опасным первым символом включаем quotePrefix
    (то же, что апостроф при ручном вводе: Excel не станет вычислять их и после правки).
    """
    value = cell.value
    if not isinstance(value, str):
        return
    cell.data_type = "s"
    if value.startswith(_FORMULA_PREFIXES):
        cell.quotePrefix = True


def append_row(ws, values) -> None:
    """`ws.append` для недоверенных данных: ни одно значение не станет формулой."""
    values = list(values)
    ws.append(values)
    if values:
        for cell in ws[ws.max_row]:
            neutralize_cell(cell)


def build_xlsx(headers: list[str], rows: list[list[str]], *, sheet_name: str = "Merged") -> bytes:
    """Собрать .xlsx из заголовков и строк (для скачивания результата)."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name[:31] or "Sheet1"
    if headers:
        append_row(ws, headers)
    for row in rows:
        append_row(ws, row)
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


# ------------------------------- 6.A.3 Различия --------------------------


def _pick_key_column(headers: list[str], a_rows: list[list[str]], b_rows: list[list[str]]) -> int | None:
    """Выбрать колонку-ключ для матчинга строк.

    Берём первую колонку, значения которой в ОБОИХ файлах уникальны и непусты.
    Если такой нет — None (тогда сравнение позиционное).
    """
    for col in range(len(headers)):
        a_vals = [r[col] for r in a_rows if col < len(r)]
        b_vals = [r[col] for r in b_rows if col < len(r)]
        if not a_vals or not b_vals:
            continue
        if any(v == "" for v in a_vals) or any(v == "" for v in b_vals):
            continue
        if len(set(a_vals)) == len(a_vals) and len(set(b_vals)) == len(b_vals):
            return col
    return None


def diff_workbooks(old: WorkbookData, new: WorkbookData) -> DiffResult:
    """Сравнить первые листы двух файлов: структура + строки/ячейки."""
    old_sheet = old.sheets[0] if old.sheets else None
    new_sheet = new.sheets[0] if new.sheets else None
    if old_sheet is None or new_sheet is None:
        raise DocumentError("Оба файла должны содержать хотя бы один лист с данными")

    old_sig = _structure_signature(old_sheet.headers)
    new_sig = _structure_signature(new_sheet.headers)
    old_set = set(old_sig)
    new_set = set(new_sig)

    added_columns = [h for h, s in zip(new_sheet.headers, new_sig) if s not in old_set]
    removed_columns = [h for h, s in zip(old_sheet.headers, old_sig) if s not in new_set]
    reordered = old_sig != new_sig and old_set == new_set
    structure_changed = bool(added_columns or removed_columns or reordered)

    truncated = False

    # Построчное сравнение делаем только по общим колонкам (по нормализованному имени).
    common = [s for s in old_sig if s in new_set]
    old_idx = {s: i for i, s in enumerate(old_sig)}
    new_idx = {s: i for i, s in enumerate(new_sig)}

    def project(row: list[str], idx_map: dict[str, int]) -> list[str]:
        out = []
        for s in common:
            i = idx_map[s]
            out.append(row[i] if i < len(row) else "")
        return out

    old_proj = [project(r, old_idx) for r in old_sheet.rows]
    new_proj = [project(r, new_idx) for r in new_sheet.rows]
    common_headers = [old_sheet.headers[old_idx[s]] for s in common]

    key_col = _pick_key_column(common_headers, old_proj, new_proj)
    added_rows: list[list[str]] = []
    removed_rows: list[list[str]] = []
    changed_cells: list[dict] = []

    def cap_reached() -> bool:
        nonlocal truncated
        if len(added_rows) + len(removed_rows) + len(changed_cells) >= MAX_DIFF_ITEMS:
            truncated = True
            return True
        return False

    if key_col is not None:
        old_by_key = {r[key_col]: r for r in old_proj}
        new_by_key = {r[key_col]: r for r in new_proj}
        for key, row in new_by_key.items():
            if key not in old_by_key and not cap_reached():
                added_rows.append(row)
        for key, row in old_by_key.items():
            if key not in new_by_key and not cap_reached():
                removed_rows.append(row)
        for key in old_by_key.keys() & new_by_key.keys():
            o, n = old_by_key[key], new_by_key[key]
            for col, header in enumerate(common_headers):
                if o[col] != n[col]:
                    if cap_reached():
                        break
                    changed_cells.append(
                        {"key": key, "column": header, "old": o[col], "new": n[col]}
                    )
    else:
        # Позиционное сравнение (нет надёжного ключа).
        for i in range(max(len(old_proj), len(new_proj))):
            o = old_proj[i] if i < len(old_proj) else None
            n = new_proj[i] if i < len(new_proj) else None
            if o is None and n is not None:
                if not cap_reached():
                    added_rows.append(n)
            elif n is None and o is not None:
                if not cap_reached():
                    removed_rows.append(o)
            elif o is not None and n is not None:
                for col, header in enumerate(common_headers):
                    if o[col] != n[col]:
                        if cap_reached():
                            break
                        changed_cells.append(
                            {"key": f"строка {i + 2}", "column": header, "old": o[col], "new": n[col]}
                        )

    return DiffResult(
        structure_changed=structure_changed,
        added_columns=added_columns,
        removed_columns=removed_columns,
        reordered=reordered,
        key_column=common_headers[key_col] if key_col is not None else None,
        added_rows=added_rows,
        removed_rows=removed_rows,
        changed_cells=changed_cells,
        truncated=truncated,
    )


# --------------------- Похожи ли файлы, чтобы их сравнивать ----------------
#
# Сравнение двух не связанных таблиц покажет «изменённым» всё подряд — пользу это не
# несёт, а выглядит как ответ. Поэтому до сравнения проверяем, что это версии одной
# таблицы или одного бланка (как и для документов в `text_compare`).

SAME_TABLE_HEADERS = 0.5  # общих заголовков колонок (доля меньшего набора) — одна таблица
SAME_TABLE_VALUES = 0.3  # или столько общих значений ячеек
SAME_FORM_VALUES = 0.4  # у бланков — общих надписей и значений
SIMILARITY_ROWS = 2000  # строк, по которым считаем общие значения


def _containment(a: set[str], b: set[str]) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def table_similarity(old: SheetData, new: SheetData) -> tuple[bool, str]:
    """Версии одной таблицы? Общие заголовки колонок либо заметная доля общих значений."""
    def headers(sheet: SheetData) -> set[str]:  # «Колонка N» — подставлены за пустой заголовок
        return {h for h in _structure_signature(sheet.headers) if h and not h.startswith("колонка ")}

    def values(sheet: SheetData) -> set[str]:
        return {_norm_header(v) for row in sheet.rows[:SIMILARITY_ROWS] for v in row if v.strip()}

    shared_headers = _containment(headers(old), headers(new))
    shared_values = _containment(values(old), values(new))
    same = shared_headers >= SAME_TABLE_HEADERS or shared_values >= SAME_TABLE_VALUES
    return same, f"общих заголовков колонок — {round(shared_headers * 100)} %, общих значений — {round(shared_values * 100)} %"


def form_similarity(old: "SheetGrid", new: "SheetGrid") -> tuple[bool, str]:
    """Версии одного бланка? Большая часть надписей и значений общая."""
    share = _containment({_norm_header(v) for v in old.cells.values()}, {_norm_header(v) for v in new.cells.values()})
    return share >= SAME_FORM_VALUES, f"общих надписей и значений — {round(share * 100)} %"


# ----------------------- 6.A.3 (форма) Сравнение бланков -----------------
#
# Таблица-реестр (строка-заголовок + строки данных) сравнивается выше в
# `diff_workbooks`. Но заявки/бланки из Этапа 7 — это НЕ таблицы, а формы:
# пары «подпись — значение» с объединёнными ячейками и без строки заголовков.
# К ним табличный diff неприменим (колонки/строки бессмысленны). Поэтому для
# форм сравниваем ячейки по координатам, а каждое отличие подписываем ближайшей
# текстовой ячейкой слева/сверху (для наших бланков подпись лежит в колонке B).


@dataclass
class SheetGrid:
    """Лист как разрежённая сетка непустых ячеек (для сравнения форм)."""

    name: str
    cells: dict[tuple[int, int], str]  # (row, col) -> значение (только непустые)
    max_row: int
    max_col: int

    @property
    def density(self) -> float:
        area = self.max_row * self.max_col
        return len(self.cells) / area if area else 0.0


@dataclass
class FormDiffItem:
    label: str  # подпись поля (ближайшая текстовая ячейка слева/сверху)
    coord: str  # координата изменённой ячейки, напр. "F16"
    op: str  # "changed" | "added" | "removed"
    old: str | None
    new: str | None


@dataclass
class FormDiffResult:
    items: list[FormDiffItem]
    truncated: bool

    @property
    def summary(self) -> dict[str, int]:
        out = {"changed": 0, "added": 0, "removed": 0}
        for it in self.items:
            out[it.op] += 1
        return out


def parse_xlsx_grid(content: bytes, filename: str, *, max_rows: int = GRID_SCAN_ROWS) -> list[SheetGrid]:
    """Прочитать листы как сетку непустых ячеек с координатами (для форм).

    Читаем не больше `max_rows` строк: формы малы, а у больших таблиц этого
    хватит, чтобы оценить плотность и не зачитывать весь реестр впустую.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover
        raise DocumentError("openpyxl не установлен на сервере") from exc

    try:
        workbook = load_workbook(io.BytesIO(content), data_only=True, read_only=True)
    except Exception as exc:  # noqa: BLE001
        raise DocumentError(f"Не удалось прочитать файл как .xlsx: {exc}") from exc

    grids: list[SheetGrid] = []
    for ws in workbook.worksheets:
        cells: dict[tuple[int, int], str] = {}
        max_r = max_c = 0
        for r_idx, row in enumerate(ws.iter_rows()):
            if r_idx >= max_rows:
                break
            for cell in row:
                v = _cell_to_str(cell.value)
                if v != "":
                    cells[(cell.row, cell.column)] = v
                    max_r = max(max_r, cell.row)
                    max_c = max(max_c, cell.column)
        grids.append(SheetGrid(name=ws.title, cells=cells, max_row=max_r, max_col=max_c))

    workbook.close()
    return grids


def is_form_like(grid: SheetGrid) -> bool:
    """Эвристика «форма vs таблица»: форма — разрежённый лист.

    У таблицы-реестра заполнено большинство ячеек прямоугольного блока, у формы
    значения раскиданы по странице среди подписей и объединённых ячеек — плотность
    низкая. Порог `FORM_DENSITY_THRESHOLD` уверенно разделяет наши заявки (~0.02-0.05)
    и реестры сотрудников (плотность близка к 1).
    """
    if grid.max_row == 0 or grid.max_col == 0:
        return False
    return grid.density < FORM_DENSITY_THRESHOLD


def _nearest_label(grid: SheetGrid, row: int, col: int) -> str:
    """Подпись для ячейки: ближайшая непустая ячейка слева в строке, иначе сверху."""
    left = [(c, v) for (r, c), v in grid.cells.items() if r == row and c < col]
    if left:
        return max(left, key=lambda t: t[0])[1]
    above = [(r, v) for (r, c), v in grid.cells.items() if c == col and r < row]
    if above:
        return max(above, key=lambda t: t[0])[1]
    return ""


def _coord(row: int, col: int) -> str:
    from openpyxl.utils import get_column_letter

    return f"{get_column_letter(col)}{row}"


def diff_forms(old: SheetGrid, new: SheetGrid) -> FormDiffResult:
    """Сравнить две формы по ячейкам с привязкой к подписям (6.A.3 для бланков)."""
    items: list[FormDiffItem] = []
    truncated = False
    for key in sorted(set(old.cells) | set(new.cells)):
        o = old.cells.get(key, "")
        n = new.cells.get(key, "")
        if o == n:
            continue
        if len(items) >= MAX_DIFF_ITEMS:
            truncated = True
            break
        row, col = key
        label = _nearest_label(old if o else new, row, col)
        op = "changed" if (o and n) else ("added" if n else "removed")
        items.append(
            FormDiffItem(label=label, coord=_coord(row, col), op=op, old=o or None, new=n or None)
        )
    return FormDiffResult(items=items, truncated=truncated)


# ============================ Word (.docx) ===============================


@dataclass
class DocxParagraph:
    text: str
    style: str  # имя стиля (Normal, Heading 1, …)
    is_heading: bool


@dataclass
class DocxTable:
    rows: list[list[str]]


@dataclass
class DocxData:
    filename: str
    paragraphs: list[DocxParagraph]
    tables: list[DocxTable]

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs if p.text)


def _doc_to_text(content: bytes) -> str:
    """Извлечь текст из старого бинарного .doc (Word 97-2003).

    python-docx читает только .docx, поэтому .doc конвертируем внешней утилитой.
    Бэкенды пробуются по доступности (`shutil.which`):
      - `antiword` — Linux/прод (ставится в Docker), быстрый, текст без таблиц;
      - `textutil` — macOS/dev (встроен в систему), `.doc` → txt.
    Таблицы и форматирование не извлекаются — осознанное упрощение: все наши
    Word-операции текстовые (сравнение, выжимка, извлечение), а .doc — легаси.
    Конвертация требует временного файла на диске (утилиты не читают stdin для
    бинарных форматов); каталог удаляется сразу после — stateless сохраняется.
    """
    with tempfile.TemporaryDirectory(prefix="doc_conv_") as tmp:
        src = os.path.join(tmp, "in.doc")
        with open(src, "wb") as f:
            f.write(content)

        antiword = shutil.which("antiword")
        if antiword:
            try:
                res = subprocess.run(
                    [antiword, src], capture_output=True, timeout=DOC_CONVERT_TIMEOUT
                )
            except subprocess.TimeoutExpired as exc:
                raise DocumentError("Конвертация .doc заняла слишком долго") from exc
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.decode("utf-8", errors="replace")
            # antiword не справился (например, это не настоящий .doc) — пробуем дальше

        textutil = shutil.which("textutil")
        if textutil:
            dst = os.path.join(tmp, "out.txt")
            try:
                res = subprocess.run(
                    [textutil, "-convert", "txt", "-encoding", "UTF-8", "-output", dst, src],
                    capture_output=True,
                    timeout=DOC_CONVERT_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                raise DocumentError("Конвертация .doc заняла слишком долго") from exc
            if res.returncode == 0 and os.path.exists(dst):
                with open(dst, "rb") as f:
                    return f.read().decode("utf-8", errors="replace")

        raise DocumentError(
            "Не удалось обработать .doc: на сервере нет конвертера (antiword/textutil). "
            "Пересохраните файл как .docx и загрузите снова."
        )


def _parse_doc_legacy(content: bytes, filename: str) -> DocxData:
    """Разобрать старый .doc: только текст (по строкам), без таблиц.

    Заголовки из plain-text надёжно не восстановить, поэтому `is_heading=False`
    у всех абзацев — извлечение покажет текст, выжимка/сравнение работают как
    обычно (они и так текстовые).
    """
    text = _doc_to_text(content)
    paragraphs = [
        DocxParagraph(text=line.strip(), style="Normal", is_heading=False)
        for line in text.splitlines()
        if line.strip()
    ]
    if not paragraphs:
        raise DocumentError("В файле .doc не найдено текста (возможно, повреждён или пуст)")
    return DocxData(filename=filename, paragraphs=paragraphs, tables=[])


def parse_docx(content: bytes, filename: str) -> DocxData:
    """Разобрать Word-файл (6.B.1).

    `.docx` — нативно через python-docx (абзацы с пометкой заголовков, таблицы).
    `.doc` (старый бинарный) — через внешний конвертер, только текст.
    """
    lower = (filename or "").lower()
    is_doc = lower.endswith(".doc") and not lower.endswith(".docx")
    if not (lower.endswith(".docx") or is_doc):
        raise DocumentError("Ожидается файл Word в формате .docx или .doc")
    _check_size(content, MAX_FILE_SIZE_WORD)

    if is_doc:
        return _parse_doc_legacy(content, filename)

    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover
        raise DocumentError("python-docx не установлен на сервере") from exc

    try:
        document = Document(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001 — python-docx кидает разнородные ошибки
        raise DocumentError(f"Не удалось прочитать файл как .docx: {exc}") from exc

    paragraphs: list[DocxParagraph] = []
    for para in document.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        style = (para.style.name if para.style else "") or "Normal"
        paragraphs.append(
            DocxParagraph(text=text, style=style, is_heading=style.lower().startswith("heading"))
        )

    tables: list[DocxTable] = []
    for table in document.tables:
        rows = [[(cell.text or "").strip() for cell in row.cells] for row in table.rows]
        tables.append(DocxTable(rows=rows))

    return DocxData(filename=filename, paragraphs=paragraphs, tables=tables)


# Сравнение двух Word-файлов (и Word с PDF) — по словам, в `text_compare`.


# ============================== PDF → Word/Excel =========================

MAX_FILE_SIZE_PDF = 20 * 1024 * 1024  # 20 МБ на PDF (между Excel и Word)
MAX_PDF_PAGES = 200  # страниц на документ; больше — отказ, а не молчаливая обрезка
OCR_LANG = "rus+eng"  # документы на русском, с латиницей в реквизитах/номерах
OCR_DPI = 200  # компромисс скорость/качество распознавания
MIN_TEXT_LAYER_CHARS = 20  # короче (и есть картинка) — считаем страницу сканом → OCR
OCR_TIMEOUT = 60  # сек на OCR одной страницы


@dataclass
class PdfTable:
    page: int
    rows: list[list[str]]
    bbox: tuple[float, float, float, float] | None = None  # область на странице


@dataclass
class PdfBlock:
    """Кусок страницы в порядке чтения: абзац текста или таблица."""

    text: str = ""  # для текстового блока
    table: PdfTable | None = None  # для таблицы


@dataclass
class PdfPageText:
    page: int  # номер страницы, с 1
    text: str  # весь текст страницы вне таблиц (для выжимки/проверок)
    is_scanned: bool  # текст получен OCR (или страница без текста вовсе)
    blocks: list[PdfBlock] = field(default_factory=list)  # текст и таблицы сверху вниз


@dataclass
class PdfData:
    filename: str
    pages: list[PdfPageText]
    tables: list[PdfTable]
    ocr_available: bool = True  # False → сканы остались без текста, т.к. нет Tesseract

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def scanned_pages(self) -> list[int]:
        return [p.page for p in self.pages if p.is_scanned]


def _ocr_page(page) -> str | None:
    """Распознать страницу-скан через Tesseract.

    Возвращает текст, либо `None`, если OCR недоступен (нет бинарника Tesseract
    или Python-обвязки) — чтобы вызывающий мог честно сказать об этом
    пользователю, а не молча отдать пустую страницу. Любая ошибка распознавания
    одной страницы не роняет разбор всего файла.
    """
    if not shutil.which("tesseract"):
        return None
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None

    try:
        pix = page.get_pixmap(dpi=OCR_DPI)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(image, lang=OCR_LANG, timeout=OCR_TIMEOUT).strip()
    except Exception:  # noqa: BLE001 — OCR best-effort: страница останется без текста
        return ""


def _extract_tables(page, page_no: int) -> list[PdfTable]:
    """Таблицы текстового слоя страницы; пустые ложные срабатывания отбрасываем."""
    try:
        found = page.find_tables()
    except Exception:  # noqa: BLE001 — детектор таблиц не должен ронять разбор
        return []

    tables: list[PdfTable] = []
    for table in found.tables:
        rows = [
            [_cell_to_str(cell).replace("\n", " ") for cell in row]
            for row in table.extract()
        ]
        rows = [r for r in rows if any(c for c in r)]
        # Таблица из одной строки/колонки — почти всегда ложное срабатывание на рамке.
        if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
            tables.append(PdfTable(page=page_no, rows=rows, bbox=tuple(table.bbox)))
    return tables


def _overlap_ratio(box: tuple[float, float, float, float], area: tuple[float, float, float, float]) -> float:
    """Какая доля прямоугольника `box` лежит внутри `area` (0..1)."""
    w = min(box[2], area[2]) - max(box[0], area[0])
    h = min(box[3], area[3]) - max(box[1], area[1])
    full = (box[2] - box[0]) * (box[3] - box[1])
    if w <= 0 or h <= 0 or full <= 0:
        return 0.0
    return (w * h) / full


# Текстовый блок считаем «внутри таблицы», только если он почти целиком в её рамке.
# Центра блока недостаточно: ложно широкая рамка таблицы иначе съела бы подписи и
# поля, лежащие рядом/под ней, — а они не должны пропадать из результата.
TABLE_OVERLAP_THRESHOLD = 0.8


def _page_blocks(page, tables: list[PdfTable]) -> list[PdfBlock]:
    """Блоки страницы сверху вниз: текст вне таблиц + сами таблицы на своих местах.

    Содержимое таблиц отдаётся только таблицей (не дублируется текстом). Значения
    заполняемых полей формы (AcroForm), если их нет в текстовом слое, дописываются
    блоками на своих местах — иначе такие поля потерялись бы.
    """
    items: list[tuple[float, float, PdfBlock]] = []  # (y0, x0, блок)

    for x0, y0, x1, y1, text, _no, block_type in page.get_text("blocks", sort=True):
        if block_type != 0 or not text.strip():
            continue
        box = (x0, y0, x1, y1)
        if any(t.bbox and _overlap_ratio(box, t.bbox) >= TABLE_OVERLAP_THRESHOLD for t in tables):
            continue
        items.append((y0, x0, PdfBlock(text=text.strip())))

    for table in tables:
        if table.bbox:
            items.append((table.bbox[1], table.bbox[0], PdfBlock(table=table)))

    page_text = page.get_text() or ""
    try:
        widgets = list(page.widgets() or [])
    except Exception:  # noqa: BLE001 — формы best-effort
        widgets = []
    for widget in widgets:
        value = _cell_to_str(widget.field_value)
        if not value or value.lower() in {"off", "false"} or value in page_text:
            continue
        label = _cell_to_str(widget.field_label or widget.field_name)
        text = f"{label}: {value}" if label else value
        items.append((widget.rect.y0, widget.rect.x0, PdfBlock(text=text)))

    items.sort(key=lambda it: (round(it[0]), it[1]))
    return [block for _y, _x, block in items]


def parse_pdf(content: bytes, filename: str) -> PdfData:
    """Разобрать PDF: текст постранично (OCR для сканов) и таблицы."""
    if not (filename or "").lower().endswith(".pdf"):
        raise DocumentError("Ожидается файл в формате .pdf")
    _check_size(content, MAX_FILE_SIZE_PDF)

    try:
        import pymupdf as fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise DocumentError("PyMuPDF не установлен на сервере") from exc

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — fitz кидает разнородные ошибки
        raise DocumentError(f"Не удалось прочитать файл как PDF: {exc}") from exc

    try:
        if doc.needs_pass:
            raise DocumentError("PDF защищён паролем — снимите защиту и загрузите снова")
        if doc.page_count == 0:
            raise DocumentError("В PDF нет страниц")
        if doc.page_count > MAX_PDF_PAGES:
            raise DocumentError(
                f"В PDF слишком много страниц ({doc.page_count}). Лимит — {MAX_PDF_PAGES}."
            )

        pages: list[PdfPageText] = []
        tables: list[PdfTable] = []
        ocr_available = True
        for index in range(doc.page_count):
            page = doc[index]
            page_no = index + 1
            text = (page.get_text() or "").strip()
            # Скан = почти нет текстового слоя И есть растровая картинка. Короткая
            # страница без картинок (штамп, пустой лист) — не повод гонять OCR.
            if len(text) >= MIN_TEXT_LAYER_CHARS or not page.get_images():
                page_tables = _extract_tables(page, page_no)
                tables.extend(page_tables)
                blocks = _page_blocks(page, page_tables)
                pages.append(
                    PdfPageText(
                        page=page_no,
                        text="\n\n".join(b.text for b in blocks if b.text),
                        is_scanned=False,
                        blocks=blocks,
                    )
                )
                continue

            recognized = _ocr_page(page)
            if recognized is None:
                ocr_available = False
                recognized = text
            blocks = [PdfBlock(text=t.strip()) for t in recognized.split("\n\n") if t.strip()]
            pages.append(
                PdfPageText(page=page_no, text=recognized, is_scanned=True, blocks=blocks)
            )
    finally:
        doc.close()

    return PdfData(filename=filename, pages=pages, tables=tables, ocr_available=ocr_available)


def _paragraph_line(text: str) -> str:
    """Строки внутри блока — переносы вёрстки PDF, а не абзацы."""
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def build_docx_from_pdf(data: PdfData) -> bytes:
    """Собрать .docx: текст и таблицы в том порядке, в каком они стоят на странице."""
    from docx import Document

    document = Document()
    for idx, page in enumerate(data.pages):
        if idx:
            document.add_page_break()
        for block in page.blocks:
            if block.table is not None:
                rows = block.table.rows
                word_table = document.add_table(rows=0, cols=max(len(r) for r in rows))
                word_table.style = "Table Grid"
                for row in rows:
                    cells = word_table.add_row().cells
                    for i, value in enumerate(row):
                        cells[i].text = value
            elif (line := _paragraph_line(block.text)):
                document.add_paragraph(line)

    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def build_xlsx_from_pdf(data: PdfData) -> bytes:
    """Собрать .xlsx: лист на страницу; текст и таблицы идут сверху вниз, как в PDF.

    Текст вне таблиц (заголовки, подписи, поля под таблицей) пишется в колонку A
    отдельными строками — иначе он терялся бы, ведь Excel «про таблицы». Если в
    документе нет ни одной таблицы — ошибка, а не Excel из одного текста.
    """
    if not data.tables:
        raise DocumentError(
            "В файле не найдено таблиц. Если нужен просто текст, "
            "попросите «переведи в Word»."
        )

    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    for page in data.pages:
        if not page.blocks:
            continue
        ws = wb.create_sheet(f"Стр{page.page}")
        last_kind: str | None = None
        for block in page.blocks:
            if block.table is not None:
                kind, lines = "table", block.table.rows
            else:
                line = _paragraph_line(block.text)
                kind, lines = "text", ([[line]] if line else [])
            if not lines:
                continue
            if last_kind is not None and last_kind != kind:
                ws.append([])  # пустая строка на границе «текст ↔ таблица»
            for row in lines:
                append_row(ws, row)
            last_kind = kind

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
