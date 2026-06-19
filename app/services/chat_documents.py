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
    MergeResult,
    WorkbookData,
    build_xlsx,
    diff_docx,
    diff_workbooks,
    merge_workbooks,
    parse_docx,
    parse_xlsx,
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


def _docx(name: str) -> bool:
    return name.lower().endswith(".docx")


def _is_supported(name: str) -> bool:
    return _xlsx(name) or _docx(name)


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
            "Поддерживаются только файлы .xlsx и .docx. "
            f"Не поддерживается: {', '.join(unsupported)}"
        )

    msg = message.strip().lower()
    xlsx_files = [(n, b) for n, b in files if _xlsx(n)]
    docx_files = [(n, b) for n, b in files if _docx(n)]

    if xlsx_files and docx_files:
        raise DocumentError("Приложите файлы одного типа: либо .xlsx, либо .docx")

    if xlsx_files:
        return _handle_xlsx(msg, xlsx_files)
    return _handle_docx(msg, docx_files, llm)


# ------------------------------- Excel -----------------------------------


def _handle_xlsx(msg: str, files: list[tuple[str, bytes]]) -> ChatFileResult:
    workbooks = [parse_xlsx(content, name) for name, content in files]

    if len(workbooks) == 1:
        return _answer_extract_xlsx(workbooks[0])

    # Несколько файлов: сравнить или объединить.
    if _has(msg, _COMPARE_KW) and len(workbooks) == 2:
        return _answer_diff_xlsx(workbooks[0], workbooks[1])
    if len(workbooks) == 2 and not _has(msg, _MERGE_KW):
        # Два файла без явного «объедини» — по умолчанию сравниваем (частый сценарий).
        return _answer_diff_xlsx(workbooks[0], workbooks[1])
    return _answer_merge_xlsx(workbooks)


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
    lines = [f"**Сравнение:** {old.filename} → {new.filename}", ""]

    if result.structure_changed:
        lines.append("**Изменения структуры:**")
        if result.added_columns:
            lines.append(f"- добавлены колонки: {', '.join(result.added_columns)}")
        if result.removed_columns:
            lines.append(f"- удалены колонки: {', '.join(result.removed_columns)}")
        if result.reordered:
            lines.append("- изменён порядок колонок")
        lines.append("")

    s = result.summary if hasattr(result, "summary") else None
    lines.append(
        f"**Итог:** добавлено строк — {len(result.added_rows)}, "
        f"удалено — {len(result.removed_rows)}, изменено ячеек — {len(result.changed_cells)}."
    )
    if result.key_column:
        lines.append(f"_Строки сопоставлены по колонке «{result.key_column}»._")
    else:
        lines.append("_Строки сопоставлены по позиции (уникальный ключ не найден)._")
    lines.append("")

    if result.added_rows:
        lines.append(f"**Добавленные строки** (до {CHAT_DIFF_ITEMS}):")
        for row in result.added_rows[:CHAT_DIFF_ITEMS]:
            lines.append(f"- {' | '.join(row)}")
        lines.append("")
    if result.removed_rows:
        lines.append(f"**Удалённые строки** (до {CHAT_DIFF_ITEMS}):")
        for row in result.removed_rows[:CHAT_DIFF_ITEMS]:
            lines.append(f"- {' | '.join(row)}")
        lines.append("")
    if result.changed_cells:
        lines.append(f"**Изменённые ячейки** (до {CHAT_DIFF_ITEMS}):")
        for c in result.changed_cells[:CHAT_DIFF_ITEMS]:
            lines.append(f"- [{c['key']}] {c['column']}: «{c['old']}» → «{c['new']}»")
        lines.append("")

    if not (result.added_rows or result.removed_rows or result.changed_cells or result.structure_changed):
        lines.append("Файлы идентичны по содержимому. ✅")
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
    lines = [
        f"**Сравнение:** {old.filename} → {new.filename}",
        "",
        f"**Итог:** изменено абзацев — {summary['changed']}, "
        f"добавлено — {summary['added']}, удалено — {summary['removed']}.",
        "",
    ]
    if not result.items:
        lines.append("Документы идентичны по тексту. ✅")
    for it in result.items[:CHAT_DIFF_ITEMS]:
        if it.op == "changed":
            lines.append(f"- ✏️ «{it.old}» → «{it.new}»")
        elif it.op == "added":
            lines.append(f"- ➕ {it.new}")
        elif it.op == "removed":
            lines.append(f"- ➖ {it.old}")
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
