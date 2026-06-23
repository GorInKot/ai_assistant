"""Обработка файлов, приложенных к сообщению в чате (Этап 6).

Пользователь прикладывает файл(ы) к сообщению и пишет, что с ними сделать —
как у обычного ИИ-ассистента. Здесь по тексту сообщения и набору файлов
выбирается операция (извлечь / выжимка / сравнить / объединить) и формируется
ответ в markdown для показа в чате.

Ядро операций — в `document_processing`; здесь только маршрутизация по намерению
и форматирование человекочитаемого ответа.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from app.llm import LLMService
from app.services.document_processing import (
    DiffResult,
    DocumentError,
    DocxData,
    FormDiffResult,
    MergeResult,
    WorkbookData,
    build_xlsx,
    diff_docx,
    diff_forms,
    diff_workbooks,
    is_form_like,
    merge_workbooks,
    parse_docx,
    parse_xlsx,
    parse_xlsx_grid,
)


CHAT_PREVIEW_ROWS = 12  # сколько строк таблицы показываем в чате
CHAT_DIFF_ITEMS = 30  # сколько элементов diff показываем в чате


@dataclass
class Attachment:
    filename: str
    content_base64: str
    mime: str


@dataclass
class ChatFileResult:
    answer: str  # markdown для показа в чате
    attachment: Attachment | None = None


# Ключевые слова намерений (нормализованный, lower-case текст сообщения).
_COMPARE_KW = ("сравн", "различ", "отлич", "разниц", "diff", "что измен")
_MERGE_KW = ("объедин", "слей", "слить", "соедини", "склей", "merge")
_SUMMARY_KW = ("выжимк", "резюме", "кратк", "саммари", "summary", "о ч[её]м", "суть", "перескаж")
_EXTRACT_KW = ("извлеки", "что в файл", "таблиц", "содерж", "распарс", "покажи данные", "разбери")


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _xlsx(name: str) -> bool:
    return name.lower().endswith(".xlsx")


def _word(name: str) -> bool:
    """Word-файл: новый .docx или старый бинарный .doc."""
    lower = name.lower()
    return lower.endswith(".docx") or lower.endswith(".doc")


def _is_supported(name: str) -> bool:
    return _xlsx(name) or _word(name)


def handle_chat_files(
    message: str,
    files: list[tuple[str, bytes]],
    llm: LLMService,
) -> ChatFileResult:
    """Главная точка входа: выбрать операцию по сообщению и файлам."""
    if not files:
        raise DocumentError("Не приложено ни одного файла")

    unsupported = [name for name, _ in files if not _is_supported(name)]
    if unsupported:
        raise DocumentError(
            "Поддерживаются только файлы .xlsx, .docx и .doc. "
            f"Не поддерживается: {', '.join(unsupported)}"
        )

    msg = message.strip().lower()
    xlsx_files = [(n, b) for n, b in files if _xlsx(n)]
    docx_files = [(n, b) for n, b in files if _word(n)]

    if xlsx_files and docx_files:
        raise DocumentError("Приложите файлы одного типа: либо Excel, либо Word")

    if xlsx_files:
        return _handle_xlsx(msg, xlsx_files)
    return _handle_docx(msg, docx_files, llm)


# ------------------------------- Excel -----------------------------------


def _handle_xlsx(msg: str, files: list[tuple[str, bytes]]) -> ChatFileResult:
    workbooks = [parse_xlsx(content, name) for name, content in files]

    if len(workbooks) == 1:
        return _answer_extract_xlsx(workbooks[0])

    # Два файла без явного «объедини» — по умолчанию сравниваем (частый сценарий).
    if len(workbooks) == 2 and (_has(msg, _COMPARE_KW) or not _has(msg, _MERGE_KW)):
        return _diff_two_xlsx(files, workbooks)
    return _answer_merge_xlsx(workbooks)


def _diff_two_xlsx(
    files: list[tuple[str, bytes]], workbooks: list[WorkbookData]
) -> ChatFileResult:
    """Сравнить два .xlsx, выбрав режим: бланк-форма или таблица-реестр."""
    grid_a = parse_xlsx_grid(files[0][1], files[0][0])
    grid_b = parse_xlsx_grid(files[1][1], files[1][0])
    a = grid_a[0] if grid_a else None
    b = grid_b[0] if grid_b else None
    if a and b and is_form_like(a) and is_form_like(b):
        return _answer_diff_form(files[0][0], files[1][0], a, b)
    return _answer_diff_xlsx(workbooks[0], workbooks[1])


def _answer_diff_form(name_old: str, name_new: str, old, new) -> ChatFileResult:
    """Человекочитаемый отчёт о различиях двух бланков (формы, не таблицы)."""
    result: FormDiffResult = diff_forms(old, new)
    lines = [f"**Различия между бланками:** «{name_old}» и «{name_new}»", ""]

    if not result.items:
        lines.append("Файлы идентичны по содержимому. ✅")
        return ChatFileResult(answer="\n".join(lines).strip())

    s = result.summary
    lines.append(
        f"**Итог:** изменено полей — {s['changed']}, "
        f"только в «{name_new}» — {s['added']}, только в «{name_old}» — {s['removed']}."
    )

    shown = result.items[:CHAT_DIFF_ITEMS]

    def _title(it) -> str:
        return f"{it.label} ({it.coord})" if it.label else it.coord

    changed = [it for it in shown if it.op == "changed"]
    only_new = [it for it in shown if it.op == "added"]
    only_old = [it for it in shown if it.op == "removed"]

    if changed:
        lines.append("")
        lines.append("**Изменённые поля:**")
        lines += [f"- **{_title(it)}**: «{it.old}» → «{it.new}»" for it in changed]
    if only_new:
        lines.append("")
        lines.append(f"**Только в документе «{name_new}»:**")
        lines += [f"- **{_title(it)}**: «{it.new}»" for it in only_new]
    if only_old:
        lines.append("")
        lines.append(f"**Только в документе «{name_old}»:**")
        lines += [f"- **{_title(it)}**: «{it.old}»" for it in only_old]

    if result.truncated or len(result.items) > CHAT_DIFF_ITEMS:
        lines.append("\n_Показаны не все различия._")
    return ChatFileResult(answer="\n".join(lines).strip())


def _answer_extract_xlsx(wb: WorkbookData) -> ChatFileResult:
    lines = [f"**Файл:** {wb.filename}", ""]
    for sheet in wb.sheets:
        lines.append(f"**Лист «{sheet.name}»** — {sheet.total_rows} строк × {sheet.n_cols} колонок")
        if sheet.truncated:
            lines.append(f"_(показаны первые {len(sheet.rows)} строк)_")
        if sheet.headers:
            lines.append("")
            lines.append(_md_table(sheet.headers, sheet.rows[:CHAT_PREVIEW_ROWS]))
            if len(sheet.rows) > CHAT_PREVIEW_ROWS:
                lines.append(f"\n_…ещё {len(sheet.rows) - CHAT_PREVIEW_ROWS} строк в файле_")
        lines.append("")
    return ChatFileResult(answer="\n".join(lines).strip())


def _answer_diff_xlsx(old: WorkbookData, new: WorkbookData) -> ChatFileResult:
    result: DiffResult = diff_workbooks(old, new)
    lines = [f"**Различия между файлами:** «{old.filename}» и «{new.filename}»", ""]

    if not (result.added_rows or result.removed_rows or result.changed_cells or result.structure_changed):
        lines.append("Файлы идентичны по содержимому. ✅")
        return ChatFileResult(answer="\n".join(lines).strip())

    lines.append(
        f"**Итог:** только в «{new.filename}» — {len(result.added_rows)}, "
        f"только в «{old.filename}» — {len(result.removed_rows)}, "
        f"изменено ячеек — {len(result.changed_cells)}."
    )
    if result.key_column:
        lines.append(f"_Строки сопоставлены по колонке «{result.key_column}»._")
    else:
        lines.append("_Строки сопоставлены по позиции (уникальный ключ не найден)._")

    if result.structure_changed:
        lines.append("")
        lines.append("**Изменения структуры:**")
        if result.added_columns:
            lines.append(f"- колонки только в «{new.filename}»: {', '.join(result.added_columns)}")
        if result.removed_columns:
            lines.append(f"- колонки только в «{old.filename}»: {', '.join(result.removed_columns)}")
        if result.reordered:
            lines.append("- изменён порядок колонок")

    if result.added_rows:
        lines.append("")
        lines.append(f"**Только в документе «{new.filename}»** (до {CHAT_DIFF_ITEMS} строк):")
        for row in result.added_rows[:CHAT_DIFF_ITEMS]:
            lines.append(f"- строка «{' | '.join(row)}»")
    if result.removed_rows:
        lines.append("")
        lines.append(f"**Только в документе «{old.filename}»** (до {CHAT_DIFF_ITEMS} строк):")
        for row in result.removed_rows[:CHAT_DIFF_ITEMS]:
            lines.append(f"- строка «{' | '.join(row)}»")
    if result.changed_cells:
        lines.append("")
        lines.append(f"**Изменённые ячейки** (до {CHAT_DIFF_ITEMS}):")
        for c in result.changed_cells[:CHAT_DIFF_ITEMS]:
            lines.append(f"- [{c['key']}] {c['column']}: «{c['old']}» → «{c['new']}»")

    if result.truncated:
        lines.append("\n_Отчёт обрезан: слишком много различий._")

    return ChatFileResult(answer="\n".join(lines).strip())


def _answer_merge_xlsx(workbooks: list[WorkbookData]) -> ChatFileResult:
    result: MergeResult = merge_workbooks(workbooks)
    if not result.ok:
        lines = ["**Объединить не удалось — структура файлов различается:**", ""]
        for m in result.mismatches:
            lines.append(f"- {m}")
        lines.append("\nПриведите колонки к единому виду и попробуйте снова.")
        return ChatFileResult(answer="\n".join(lines))

    xlsx_bytes = build_xlsx(result.headers, result.rows)
    attachment = Attachment(
        filename="merged.xlsx",
        content_base64=base64.b64encode(xlsx_bytes).decode("ascii"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    lines = [
        f"**Объединено файлов:** {len(workbooks)} → **{result.total_rows} строк**.",
        "",
        "Вклад каждого файла:",
    ]
    for fn, n in result.source_counts:
        lines.append(f"- {fn}: {n} строк")
    lines.append("\nГотовый файл `merged.xlsx` ниже — можно скачать. 👇")
    return ChatFileResult(answer="\n".join(lines), attachment=attachment)


# -------------------------------- Word -----------------------------------


def _handle_docx(msg: str, files: list[tuple[str, bytes]], llm: LLMService) -> ChatFileResult:
    docs = [parse_docx(content, name) for name, content in files]

    if len(docs) >= 2:
        if len(docs) > 2:
            raise DocumentError("Сравнение поддерживает ровно два .docx-файла")
        return _answer_diff_docx(docs[0], docs[1])

    doc = docs[0]
    # Один файл: выжимка по умолчанию; «извлеки/таблицы» → структура.
    if _has(msg, _EXTRACT_KW) and not _has(msg, _SUMMARY_KW):
        return _answer_extract_docx(doc)
    return _answer_summary_docx(doc, llm)


def _answer_extract_docx(doc: DocxData) -> ChatFileResult:
    lines = [
        f"**Файл:** {doc.filename}",
        f"Абзацев: {len(doc.paragraphs)}, таблиц: {len(doc.tables)}",
        "",
    ]
    headings = [p.text for p in doc.paragraphs if p.is_heading][:20]
    if headings:
        lines.append("**Заголовки:**")
        for h in headings:
            lines.append(f"- {h}")
        lines.append("")
    preview = [p.text for p in doc.paragraphs if not p.is_heading][:5]
    if preview:
        lines.append("**Начало текста:**")
        for p in preview:
            lines.append(f"> {p}")
    return ChatFileResult(answer="\n".join(lines).strip())


def _answer_summary_docx(doc: DocxData, llm: LLMService) -> ChatFileResult:
    text = doc.full_text
    if not text.strip():
        raise DocumentError("В документе нет текста для выжимки")
    from app.services.document_processing import MAX_SUMMARY_CHARS

    summary = llm.summarize_text(text[:MAX_SUMMARY_CHARS])
    note = "\n\n_Документ длинный — выжимка по началу текста._" if len(text) > MAX_SUMMARY_CHARS else ""
    return ChatFileResult(answer=f"**Выжимка — {doc.filename}:**\n\n{summary}{note}")


def _answer_diff_docx(old: DocxData, new: DocxData) -> ChatFileResult:
    result = diff_docx(old, new)
    summary = result.summary
    lines = [f"**Различия между файлами:** «{old.filename}» и «{new.filename}»", ""]

    if not result.items:
        lines.append("Документы идентичны по тексту. ✅")
        return ChatFileResult(answer="\n".join(lines).strip())

    lines.append(
        f"**Итог:** только в «{new.filename}» — {summary['added']}, "
        f"только в «{old.filename}» — {summary['removed']}, изменено — {summary['changed']}."
    )

    shown = result.items[:CHAT_DIFF_ITEMS]
    only_new = [it.new for it in shown if it.op == "added"]
    only_old = [it.old for it in shown if it.op == "removed"]
    changed = [(it.old, it.new) for it in shown if it.op == "changed"]

    # Каждое отличие привязано к документу, в котором оно есть.
    if only_new:
        lines.append("")
        lines.append(f"**Только в документе «{new.filename}»:**")
        lines += [f"- строка «{t}»" for t in only_new]
    if only_old:
        lines.append("")
        lines.append(f"**Только в документе «{old.filename}»:**")
        lines += [f"- строка «{t}»" for t in only_old]
    if changed:
        lines.append("")
        lines.append("**Изменённые строки:**")
        lines += [f"- «{o}» → «{n}»" for o, n in changed]

    if result.truncated or len(result.items) > CHAT_DIFF_ITEMS:
        lines.append("\n_Показаны не все различия._")
    return ChatFileResult(answer="\n".join(lines).strip())


# ----------------------------- markdown helpers --------------------------


def _md_cell(value: str) -> str:
    v = (value or "").replace("|", "\\|").replace("\n", " ")
    return v[:60] + "…" if len(v) > 60 else v


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Таблица как fenced code-block (моноширинно), т.к. во фронте нет remark-gfm.

    Колонки выравниваются по ширине — читается аккуратно в любом md-рендере.
    """
    cols = [_md_cell(h) for h in headers]
    matrix = [cols] + [[_md_cell(c) for c in row] for row in rows]
    widths = [0] * len(cols)
    for r in matrix:
        for i in range(len(cols)):
            cell = r[i] if i < len(r) else ""
            widths[i] = max(widths[i], len(cell))
    lines = []
    for r_idx, r in enumerate(matrix):
        cells = [(r[i] if i < len(r) else "").ljust(widths[i]) for i in range(len(cols))]
        lines.append("  ".join(cells).rstrip())
        if r_idx == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(cols))))
    return "```\n" + "\n".join(lines) + "\n```"
