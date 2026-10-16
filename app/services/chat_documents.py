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
import copy
import hashlib
import re
import threading
import time
from dataclasses import asdict, dataclass, field

from app.llm import LLMService
from app.services import doc_identify, registry, spec_compare, spec_registry, text_compare
from app.services.document_processing import (
    DiffResult,
    DocumentError,
    DocxData,
    FormDiffResult,
    MergeResult,
    PdfData,
    WorkbookData,
    build_docx_from_pdf,
    build_xlsx,
    build_xlsx_from_pdf,
    diff_forms,
    diff_workbooks,
    form_similarity,
    is_form_like,
    merge_workbooks,
    parse_docx,
    parse_pdf,
    parse_xlsx,
    parse_xlsx_grid,
    table_similarity,
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
    # Предложенные действия с тем же файлом (кнопки в интерфейсе): {label, message}
    suggestions: list[dict] = field(default_factory=list)


# Ключевые слова намерений (нормализованный, lower-case текст сообщения).
_COMPARE_KW = ("сравн", "различ", "отлич", "разниц", "diff", "что измен")
_MERGE_KW = ("объедин", "слей", "слить", "соедини", "склей", "merge")
_SUMMARY_KW = ("выжимк", "резюме", "кратк", "саммари", "summary", "о чём", "о чем", "суть", "перескаж")
_EXCEL_KW = ("excel", "эксель", "xlsx", "таблиц")
_REGISTRY_KW = ("реестр", "извлек", "извлеч", "выпиши", "вытащи", "достань", "собери данные", "реквизит", "поля")
_SPEC_KW = ("спецификац",)  # реестр позиций по форме ГОСТ 21.110 (сканы), без модели
_POSITIONS_KW = ("позици",)
_WORD_KW = ("word", "ворд", "docx")
_ANYWAY_KW = ("всё равно", "все равно", "всё-таки", "все-таки")
_CONVERT_KW = ("word", "ворд", "docx", "конверт", "переведи", "перевести") + _EXCEL_KW
_EXTRACT_KW = ("извлеки", "что в файл", "таблиц", "содерж", "распарс", "покажи данные", "разбери")
# Файл без указаний или вопрос «что это» — определить вид документа и предложить действия.
_WHAT_KW = ("что это", "что за файл", "что за документ", "что можно", "что можешь", "что с ним", "что с этим",
            "посмотри", "глянь", "определи", "какой это", "какой документ")


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _xlsx(name: str) -> bool:
    return name.lower().endswith(".xlsx")


def _word(name: str) -> bool:
    """Word-файл: новый .docx или старый бинарный .doc."""
    lower = name.lower()
    return lower.endswith(".docx") or lower.endswith(".doc")


def _pdf(name: str) -> bool:
    return name.lower().endswith(".pdf")


def _is_supported(name: str) -> bool:
    return _xlsx(name) or _word(name) or _pdf(name)


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
            "Поддерживаются только файлы .xlsx, .docx, .doc и .pdf. "
            f"Не поддерживается: {', '.join(unsupported)}"
        )

    msg = message.strip().lower()
    xlsx_files = [(n, b) for n, b in files if _xlsx(n)]
    docx_files = [(n, b) for n, b in files if _word(n)]
    pdf_files = [(n, b) for n, b in files if _pdf(n)]

    if len(files) == 1 and not xlsx_files and (not msg or _has(msg, _WHAT_KW)):
        return _handle_identify(files[0], llm)

    if sum(bool(group) for group in (xlsx_files, docx_files, pdf_files)) > 1:
        if len(files) == 2 and not xlsx_files:
            # PDF и Word — сравнить по тексту (например, скан подписанного документа и исходник)
            return _handle_text_compare(msg, files)
        raise DocumentError("Приложите файлы одного типа: Excel, Word или PDF")

    if pdf_files:
        return _handle_pdf(msg, pdf_files, llm)

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
        return _diff_two_xlsx(msg, files, workbooks)
    return _answer_merge_xlsx(workbooks)


def _diff_two_xlsx(
    msg: str, files: list[tuple[str, bytes]], workbooks: list[WorkbookData]
) -> ChatFileResult:
    """Сравнить два .xlsx, выбрав режим: бланк-форма или таблица-реестр.

    Старый файл — по именам файлов или по указанию пользователя, иначе ассистент спрашивает.
    Не похожие друг на друга файлы не сравниваются (если не попросили «сравни всё равно»)."""
    names = [name for name, _ in files]
    ordered_files, chosen, ask = _version_order(msg, files)
    if ordered_files[0] is not files[0]:
        workbooks = workbooks[::-1]
    files = ordered_files
    swap, _note = spec_compare.order_files(files[0][0], files[1][0], chosen=chosen)
    if swap:
        files, workbooks = files[::-1], workbooks[::-1]
    force = _has(msg, _ANYWAY_KW)
    grid_a = parse_xlsx_grid(files[0][1], files[0][0])
    grid_b = parse_xlsx_grid(files[1][1], files[1][0])
    a = grid_a[0] if grid_a else None
    b = grid_b[0] if grid_b else None
    if a and b and is_form_like(a) and is_form_like(b):
        same, evidence = form_similarity(a, b)
        if not same and not force:
            return _not_similar(files, "одного бланка", evidence)
        if ask:
            return _ask_old_version(msg, names)
        return _answer_diff_form(files[0][0], files[1][0], a, b)
    if workbooks[0].sheets and workbooks[1].sheets:
        same, evidence = table_similarity(workbooks[0].sheets[0], workbooks[1].sheets[0])
        if not same and not force:
            return _not_similar(files, "одной таблицы", evidence)
    if ask:
        return _ask_old_version(msg, names)
    return _answer_diff_xlsx(workbooks[0], workbooks[1])


_OLD_RE = re.compile(r"\bстар")
_NEW_RE = re.compile(r"\bнов")


def _chosen_old(msg: str, names: list[str]) -> int | None:
    """Какую версию пользователь назвал старой: «старая — «имя файла»», «новая — …»,
    «старая — первая / вторая». Индекс старой или None, если не сказано однозначно."""
    if _OLD_RE.search(msg):
        named_old = True
    elif _NEW_RE.search(msg):
        named_old = False
    else:
        return None
    low = [n.lower() for n in names]
    hit = None
    if low[0] != low[1]:
        for pattern in ("«{}»", "{}"):
            hits = [i for i, n in enumerate(low) if pattern.format(n) in msg]
            if len(hits) == 1:
                hit = hits[0]
                break
    if hit is None:
        first, second = "перв" in msg, "втор" in msg
        if first == second:
            return None
        hit = 0 if first else 1
    return hit if named_old else 1 - hit


def _short_name(name: str, limit: int = 40) -> str:
    return name if len(name) <= limit else f"{name[:limit - 12]}…{name[-11:]}"


def _ask_old_version(msg: str, names: list[str]) -> ChatFileResult:
    """По именам не понять, какая версия старая, — спросить, а не угадывать по порядку."""
    base = msg if _has(msg, _COMPARE_KW) else f"сравни {msg}".strip()
    if names[0].lower() == names[1].lower():
        options = [("первый приложенный файл", "старая — первый файл"),
                   ("второй приложенный файл", "старая — второй файл")]
    else:
        options = [(_short_name(n), f"старая — «{n}»") for n in names]
    suggestions = [doc_identify.Suggestion(f"Старая — {label}", f"{base} ({choice})") for label, choice in options]
    return ChatFileResult(
        answer=(
            "**Какая версия старая?** По именам файлов это не определить: нет номера редакции "
            "(R, rC, изм., ред.), дат или слов «старый / новый». Угадывать по порядку приложения не буду — "
            "от этого зависит, что попадёт в «добавлено», а что в «удалено».\n\n"
            "Выберите кнопкой или напишите, например: «сравни, старая — первая»."
        ),
        suggestions=[asdict(s) for s in suggestions],
    )


def _version_order(
    msg: str, files: list[tuple[str, bytes]]
) -> tuple[list[tuple[str, bytes]], bool, bool]:
    """Порядок двух версий: (файлы — указанная старая первой, указал ли её пользователь,
    нужно ли спросить). Если порядок виден по именам, его наводит само сравнение. Спрашивать
    стоит только после проверки, что это версии одного документа."""
    names = [name for name, _ in files]
    if spec_compare.order_by_names(names[0], names[1]) is not None:
        return files, False, False
    chosen = _chosen_old(msg, names)
    if chosen is None:
        return files, False, True
    return (files if chosen == 0 else files[::-1]), True, False


# Разобранные документы держим недолго: после ответа «какая версия старая» сканы не
# распознаются заново. Ключ — имя и хэш содержимого; копия на выдаче, чтобы сравнение
# не меняло сохранённое.
_PARSED_TTL = 15 * 60
_PARSED_MAX = 8
_parsed: dict[tuple[str, str, str], tuple[float, object]] = {}
_parsed_lock = threading.Lock()


def _parsed_once(kind: str, name: str, content: bytes, load):
    key = (kind, name, hashlib.sha256(content).hexdigest())
    now = time.time()
    with _parsed_lock:
        hit = _parsed.get(key)
        if hit and hit[0] > now:
            return copy.deepcopy(hit[1])
    value = load()
    with _parsed_lock:
        for old_key in [k for k, (expires, _) in _parsed.items() if expires <= now]:
            del _parsed[old_key]
        while len(_parsed) >= _PARSED_MAX:
            del _parsed[next(iter(_parsed))]
        _parsed[key] = (now + _PARSED_TTL, copy.deepcopy(value))
    return value


def _not_similar(files: list[tuple[str, bytes]], what: str, evidence: str) -> ChatFileResult:
    return ChatFileResult(answer=(
        f"Похоже, «{files[0][0]}» и «{files[1][0]}» — не версии {what} ({evidence}). "
        "Сравнение покажет изменённым почти всё, поэтому его не делаю.\n\n"
        "Если это всё-таки версии одного файла, напишите «сравни всё равно»."
    ))


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


# ------------------------------ Что за файл -------------------------------


def _handle_identify(file: tuple[str, bytes], llm: LLMService) -> ChatFileResult:
    """Файл без указаний: назвать вид документа, показать основание и предложить действия."""
    name, content = file
    found = doc_identify.identify(content, name, llm)
    lines = []
    if found.kind:
        lines.append(f"**«{name}»** — похоже на: **{found.kind}** ({found.basis}).")
    else:
        lines.append(f"**«{name}»** — вид документа определить не удалось: заголовка с видом документа "
                     "в начале нет, а модель не дала уверенного ответа.")
    title = []
    if found.title and found.title.lower() != found.kind.lower():
        title.append(f"заголовок «{found.title}»")
    if found.number:
        title.append(f"№ {found.number}")
    if found.date:
        title.append(f"от {found.date}")
    for part in (title, found.details):  # первая буква заглавная, остальное как есть («PDF», «№»)
        if part:
            text = ", ".join(part)
            lines.append(text[:1].upper() + text[1:] + ".")
    lines += ["", "Что можно сделать — выберите действие ниже или напишите своими словами:"]
    lines += [f"- {s.label}" for s in found.suggestions]
    if found.hint:
        lines += ["", found.hint]
    return ChatFileResult(answer="\n".join(lines), suggestions=[asdict(s) for s in found.suggestions])


# -------------------------------- Word -----------------------------------


def _handle_docx(msg: str, files: list[tuple[str, bytes]], llm: LLMService) -> ChatFileResult:
    if len(files) >= 2:
        if len(files) > 2:
            raise DocumentError("Сравнение поддерживает ровно два Word-файла")
        return _handle_text_compare(msg, files)

    doc = parse_docx(files[0][1], files[0][0])
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


CHAT_TEXT_CHANGES = 15  # сколько изменений текста показываем в чате (все — в файле)


def _handle_text_compare(msg: str, files: list[tuple[str, bytes]]) -> ChatFileResult:
    """Две версии документа (PDF — текст или скан, Word) — что изменилось в тексте."""
    return _text_compare_ordered(msg, files, *_version_order(msg, files))


def _text_compare_ordered(
    msg: str, original: list[tuple[str, bytes]], files: list[tuple[str, bytes]], chosen: bool, ask: bool
) -> ChatFileResult:
    started = time.time()
    docs = [
        _parsed_once("text", name, content, lambda c=content, n=name: text_compare.load_document(c, n))
        for name, content in files
    ]
    result = text_compare.compare(docs[0], docs[1], force=_has(msg, _ANYWAY_KW), chosen=chosen)
    result.seconds = round(time.time() - started, 1)
    if not result.same_document and not _has(msg, _ANYWAY_KW):
        return ChatFileResult(answer=(
            f"Похоже, «{result.old.filename}» и «{result.new.filename}» — не версии одного документа: "
            f"общего текста всего {round(result.shared * 100)} %. Сравнение здесь покажет почти весь текст "
            "как изменённый, поэтому его не делаю.\n\n"
            "Если это всё-таки версии одного документа, напишите «сравни всё равно»."
        ))
    if ask:
        return _ask_old_version(msg, [name for name, _ in original])
    answer = _text_compare_summary(result)
    if not result.changes:
        return ChatFileResult(answer=answer)
    attachment = Attachment(
        filename="comparison.docx",
        content_base64=base64.b64encode(text_compare.build_compare_docx(result)).decode("ascii"),
        mime=_DOCX_MIME,
    )
    return ChatFileResult(answer=answer + "\n\nОтчёт со всеми изменениями — в файле ниже. 👇", attachment=attachment)


def _md_text(words: list) -> str:
    return " ".join(w.text for w in words).replace("*", "\\*").replace("_", "\\_")


def _fragment(change, words: list) -> str:
    before, after = _md_text(change.before[-3:]), _md_text(change.after[:3])
    core = f"**{_md_text(words)}**" if words else "—"
    return "«" + " ".join(part for part in (before and "…" + before, core, after and after + "…") if part) + "»"


def _text_compare_summary(result: "text_compare.TextComparison") -> str:
    tc = text_compare
    lines = [
        f"**Сравнение документов** (за {int(result.seconds)} с).",
        f"Старая: «{result.old.filename}», новая: «{result.new.filename}» ({result.order_note}).",
    ]
    shared = round(result.shared * 100)
    if result.same_document:
        lines.append(f"Общий текст — {shared} %: версии одного документа.")
    else:
        lines.append(f"⚠️ Общий текст — всего {shared} %: не похоже на версии одного документа, сравнение по вашей просьбе.")
    scans = [d for d in (result.old, result.new) if d.scanned_pages]
    if scans:
        lines.append(
            "Сканы распознаны автоматически: "
            + "; ".join(f"«{d.filename}» — стр. {_page_ranges(d.scanned_pages)}" for d in scans)
            + ". Различия, похожие на ошибки распознавания, отмечены «проверить»."
        )
    if not result.changes:
        lines.append("\nРазличий в тексте нет. ✅")
        return "\n".join(lines)
    lines += [
        "",
        f"- Изменено: **{result.count(tc.STATUS_CHANGED)}**, добавлено: **{result.count(tc.STATUS_ADDED)}**, "
        f"удалено: **{result.count(tc.STATUS_REMOVED)}**.",
    ]
    if result.count(tc.STATUS_CHECK):
        lines.append(f"- ⚠️ Проверить: **{result.count(tc.STATUS_CHECK)}** — на скане прочитано неуверенно "
                     "или похоже на ошибку распознавания: сверьте со сканом.")
    lines += ["", "**Изменения:**"]
    for change in result.changes[:CHAT_TEXT_CHANGES]:
        if change.status == tc.STATUS_ADDED:
            text = f"добавлено {_fragment(change, change.new)}"
        elif change.status == tc.STATUS_REMOVED:
            text = f"удалено {_fragment(change, change.old)}"
        else:
            text = f"{_fragment(change, change.old)} → {_fragment(change, change.new)}"
        where = change.where_old if change.status == tc.STATUS_REMOVED else change.where_new
        mark = " ⚠️ проверить" if change.status == tc.STATUS_CHECK else ""
        lines.append(f"- {where}: {text}{mark}")
    if len(result.changes) > CHAT_TEXT_CHANGES:
        lines.append(f"\n_…ещё {len(result.changes) - CHAT_TEXT_CHANGES} изменений в отчёте._")
    return "\n".join(lines)


# --------------------------------- PDF -----------------------------------

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _handle_pdf(msg: str, files: list[tuple[str, bytes]], llm: LLMService) -> ChatFileResult:
    """PDF: реестр полей из нескольких документов либо конвертация одного в Word/Excel.

    Реестр — если просят «реестр/извлеки поля» или приложено несколько PDF без просьбы
    о конвертации. Один PDF без таких слов — конвертация (Word по умолчанию, Excel для таблиц).
    Про спецификации («реестр спецификаций», «позиции спецификации») — реестр позиций по форме.
    Два PDF и «сравни» — сравнение редакций спецификации по позициям.
    """
    if len(files) == 2 and _has(msg, _COMPARE_KW):
        return _handle_pdf_compare(msg, files)
    wants_registry = _has(msg, _REGISTRY_KW)
    if not _has(msg, _WORD_KW) and (_has(msg, _SPEC_KW) or (wants_registry and _has(msg, _POSITIONS_KW))):
        return _handle_spec_registry(files)
    if len(files) > 1 and not wants_registry:
        if _has(msg, _CONVERT_KW):
            raise DocumentError(
                "Конвертация в Word/Excel — по одному PDF за раз. Чтобы собрать данные из нескольких "
                "документов в один Excel, напишите, например: «составь реестр: номер, дата, сумма»."
            )
        wants_registry = True
    if len(files) == 1 and not wants_registry and _has(msg, _SUMMARY_KW):
        return _answer_summary_pdf(files[0], llm)
    if wants_registry:
        return _handle_pdf_registry(msg, files, llm)

    name, content = files[0]
    data = parse_pdf(content, name)

    if not data.full_text.strip() and not data.tables:
        if data.scanned_pages and not data.ocr_available:
            raise DocumentError(
                "Файл — скан, а распознавание текста (OCR) на сервере недоступно. "
                "Приложите PDF с текстовым слоем."
            )
        raise DocumentError("В PDF не удалось найти текст (страницы пусты или не распознаны)")

    stem = name.rsplit(".", 1)[0]
    if _has(msg, _EXCEL_KW):
        file_bytes, ext, mime = build_xlsx_from_pdf(data), "xlsx", _XLSX_MIME
    else:
        file_bytes, ext, mime = build_docx_from_pdf(data), "docx", _DOCX_MIME

    attachment = Attachment(
        filename=f"{stem}.{ext}",
        content_base64=base64.b64encode(file_bytes).decode("ascii"),
        mime=mime,
    )
    return ChatFileResult(answer=_pdf_summary(data, attachment.filename), attachment=attachment)


def _answer_summary_pdf(file: tuple[str, bytes], llm: LLMService) -> ChatFileResult:
    """Выжимка PDF (сканы распознаются) — тем же способом, что выжимка Word."""
    from app.services.document_processing import MAX_SUMMARY_CHARS

    name, content = file
    data = parse_pdf(content, name)
    text = data.full_text
    if not text.strip():
        raise DocumentError("В PDF не удалось найти текст для выжимки (страницы пусты или не распознаны)")
    summary = llm.summarize_text(text[:MAX_SUMMARY_CHARS])
    notes = []
    if len(text) > MAX_SUMMARY_CHARS:
        notes.append("_Документ длинный — выжимка по началу текста._")
    if data.scanned_pages:
        notes.append(f"_Страницы {_page_ranges(data.scanned_pages)} — сканы, текст распознан автоматически._")
    return ChatFileResult(answer=f"**Выжимка — {name}:**\n\n{summary}" + "".join("\n\n" + n for n in notes))


def _pdf_summary(data: PdfData, out_name: str) -> str:
    lines = [
        f"**Конвертировано:** «{data.filename}» → `{out_name}`",
        f"Страниц: {len(data.pages)}, найдено таблиц: {len(data.tables)}.",
    ]
    scanned = data.scanned_pages
    if scanned and data.ocr_available:
        lines.append(
            f"\n⚠️ Страницы {_page_list(scanned)} — сканы: текст распознан автоматически (OCR), "
            "возможны ошибки. Проверьте результат перед использованием."
        )
    elif scanned:
        lines.append(
            f"\n⚠️ Страницы {_page_list(scanned)} — сканы, а распознавание (OCR) на сервере "
            "недоступно: эти страницы в файле пустые."
        )
    lines.append("\nГотовый файл ниже — можно скачать. 👇")
    return "\n".join(lines)


def _page_list(pages: list[int], limit: int = 15) -> str:
    shown = ", ".join(str(p) for p in pages[:limit])
    return shown + (f" и ещё {len(pages) - limit}" if len(pages) > limit else "")


def _page_ranges(pages: list[int]) -> str:
    """[2, 4, 5, 6, 9] → «2, 4–6, 9»."""
    parts: list[str] = []
    for page in sorted(pages):
        if parts and page == int(parts[-1].split("–")[-1]) + 1:
            parts[-1] = parts[-1].split("–")[0] + f"–{page}"
        else:
            parts.append(str(page))
    return ", ".join(parts)


def _handle_pdf_registry(msg: str, files: list[tuple[str, bytes]], llm: LLMService) -> ChatFileResult:
    """Реестр: извлечь заданные поля из каждого PDF в одну таблицу Excel."""
    if len(files) > registry.MAX_REGISTRY_DOCS:
        raise DocumentError(
            f"Слишком много документов для одного реестра ({len(files)}). "
            f"Лимит — {registry.MAX_REGISTRY_DOCS}: модель работает локально и разбирает документ "
            "за десятки секунд. Разбейте на несколько запросов."
        )
    fields = registry.parse_fields_from_message(msg)
    docs = [parse_pdf(content, name) for name, content in files]
    result = registry.build_registry(docs, fields, llm)

    attachment = Attachment(
        filename="registry.xlsx",
        content_base64=base64.b64encode(registry.build_registry_xlsx(result)).decode("ascii"),
        mime=_XLSX_MIME,
    )
    return ChatFileResult(answer=_registry_summary(result), attachment=attachment)


def _registry_summary(result: "registry.RegistryResult") -> str:
    ok_rows = [r for r in result.rows if not r.error]
    lines = [
        f"**Реестр по {len(result.rows)} документам** → `registry.xlsx` "
        f"(разобрано за {int(result.seconds)} с).",
        f"Поля: {', '.join(result.fields)}.",
    ]
    if result.default_fields_used:
        lines.append(
            "_Поля вы не указали — взят стандартный набор. Чтобы задать свои, напишите, например: "
            "«составь реестр: номер, дата, подрядчик, сумма»._"
        )
    lines.append(f"Заполнено ячеек: {result.filled_cells}.")

    if result.unverified_cells:
        lines.append(
            f"\n⚠️ **Проверьте:** {result.unverified_cells} значений выделены жёлтым — модель выдала их, "
            "но дословно в тексте документа они не найдены (возможна ошибка или пересказ)."
        )
    scanned = [r.filename for r in ok_rows if r.scanned]
    if scanned:
        lines.append(
            f"\n⚠️ Сканы, распознанные автоматически (OCR), возможны ошибки в тексте: {', '.join(scanned)}."
        )
    failed = [r for r in result.rows if r.error]
    if failed:
        lines.append("\n❌ **Не разобраны:**")
        lines += [f"- {r.filename}: {r.error}" for r in failed]

    if ok_rows:
        lines.append("")
        lines.append(_md_table(["Файл", *result.fields], [[r.filename, *[r.values[f] for f in result.fields]] for r in ok_rows[:5]]))
        if len(ok_rows) > 5:
            lines.append(f"\n_…ещё {len(ok_rows) - 5} строк в файле_")
    lines.append("\nГотовый файл ниже — можно скачать. 👇")
    return "\n".join(lines)


def _handle_spec_registry(files: list[tuple[str, bytes]]) -> ChatFileResult:
    """Реестр позиций спецификаций по сканам: форма ГОСТ 21.110, OCR по графам, без модели."""
    if len(files) > spec_registry.MAX_SPEC_FILES:
        raise DocumentError(
            f"Слишком много файлов для одного реестра ({len(files)}), лимит — {spec_registry.MAX_SPEC_FILES}."
        )
    problem = spec_registry.unavailable_reason()
    if problem:
        raise DocumentError(problem)
    for name, content in files:
        spec_registry.check_pdf(content, name)
    result = spec_registry.build_spec_registry(files)
    attachment = Attachment(
        filename="specifications.xlsx",
        content_base64=base64.b64encode(spec_registry.build_spec_xlsx(result)).decode("ascii"),
        mime=_XLSX_MIME,
    )
    return ChatFileResult(answer=_spec_summary(result), attachment=attachment)


def _spec_summary(result: "spec_registry.SpecRegistry") -> str:
    docs = result.documents
    pages = sum(d.table_pages for d in docs)
    lines = [
        f"**Реестр позиций спецификаций** → `specifications.xlsx`: {len(docs)} файл(ов), "
        f"{pages} страниц с таблицей, разобрано за {int(result.seconds)} с.",
        f"Позиций: **{result.positions}**.",
    ]
    if result.changed or result.excluded:
        lines.append(
            f"Учтены правки по документу: в {result.changed} позициях зачёркнутое значение заменено новым "
            f"(голубые, старое — в колонке «Правки по документу»), {result.excluded} позиций зачёркнуты целиком "
            "(серые, в суммы не включать)."
        )
    if result.flagged:
        lines.append(
            f"\n⚠️ **Проверьте {result.flagged}** — выделены жёлтым, причина в колонке «Проверить»: "
            "правки от руки, цветные пометки, неуверенно прочитанные числа, не число в количестве. "
            "Значения не додумываются: что не прочиталось уверенно, оставлено как прочитано."
        )
    no_form = [d for d in docs if not d.error and d.table_pages == 0]
    if no_form:
        lines.append("\nНе похожи на спецификацию по ГОСТ 21.110 (таблица из 9 граф не найдена): "
                     + ", ".join(d.filename for d in no_form) + ".")
    partial = [d for d in docs if d.table_pages and d.skipped_pages]
    if partial:
        lines.append("Страницы без таблицы спецификации (титул, лист регистрации изменений и т.п.): "
                     + "; ".join(f"{d.filename} — стр. {_page_list(d.skipped_pages, 8)}" for d in partial) + ".")
    failed = [d for d in docs if d.error]
    if failed:
        lines.append("\n❌ **Не разобраны:**")
        lines += [f"- {d.filename}: {d.error}" for d in failed]
    broken = [d for d in docs if d.failed_pages]
    if broken:
        lines.append("\n❌ Сбой распознавания отдельных страниц (остальные разобраны): "
                     + "; ".join(f"{d.filename} — стр. {_page_list(d.failed_pages, 8)}" for d in broken) + ".")
    rows = [
        [d.filename, str(d.table_pages), str(len(d.positions)), str(d.flagged)]
        for d in docs if d.table_pages
    ]
    if rows:
        lines.append("")
        lines.append(_md_table(["Файл", "Стр.", "Позиций", "Проверить"], rows[:10]))
        if len(rows) > 10:
            lines.append(f"\n_…ещё {len(rows) - 10} документов в файле_")
    lines.append("\nГотовый файл ниже — можно скачать. 👇")
    return "\n".join(lines)


def _handle_pdf_compare(msg: str, files: list[tuple[str, bytes]]) -> ChatFileResult:
    """Два PDF: если оба — спецификации по форме ГОСТ 21.110, сравнение по позициям,
    иначе — по тексту. Какая версия старая, спрашиваем после проверки, что это версии
    одного документа; разобранное не распознаётся повторно после ответа."""
    original = files
    files, chosen, ask = _version_order(msg, files)
    if spec_registry.unavailable_reason() is None:
        for name, content in files:
            spec_registry.check_pdf(content, name)
        budget = spec_registry.MAX_SPEC_PAGES // 2
        docs = [
            _parsed_once(f"spec{budget}", name, content,
                         lambda c=content, n=name: spec_registry.process_pdf(c, n, budget))
            for name, content in files
        ]
        if all(not d.error and d.table_pages for d in docs):
            return _answer_spec_compare(msg, docs, chosen, [name for name, _ in original] if ask else None)
    return _text_compare_ordered(msg, original, files, chosen, ask)


def _answer_spec_compare(
    msg: str, docs: list["spec_registry.SpecDocument"], chosen: bool = False, ask_names: list[str] | None = None
) -> ChatFileResult:
    """Две редакции спецификации: что изменилось по позициям. ask_names — порядок не известен:
    если это редакции одного документа, спросить, какая старая."""
    result = spec_compare.compare(*docs, chosen=chosen)
    result.seconds = round(sum(d.seconds for d in docs) + result.seconds, 1)
    if not result.same_document and not _has(msg, _ANYWAY_KW):
        return ChatFileResult(answer=(
            f"Похоже, «{result.old.filename}» и «{result.new.filename}» — не версии одного документа "
            f"({result.evidence}). Сравнение по позициям здесь ничего не покажет, поэтому его не делаю.\n\n"
            "Если это всё-таки редакции одного документа, напишите «сравни всё равно»."
        ))
    if ask_names:
        return _ask_old_version(msg, ask_names)
    attachment = Attachment(
        filename="spec_comparison.xlsx",
        content_base64=base64.b64encode(spec_compare.build_compare_xlsx(result)).decode("ascii"),
        mime=_XLSX_MIME,
    )
    return ChatFileResult(answer=_compare_summary(result), attachment=attachment)


def _compare_summary(result: "spec_compare.SpecComparison") -> str:
    sc = spec_compare
    lines = [
        f"**Сравнение редакций спецификации** → `spec_comparison.xlsx` (разобрано за {int(result.seconds)} с).",
        f"Старая: «{result.old.filename}», новая: «{result.new.filename}» ({result.order_note}).",
    ]
    if result.same_document:
        lines.append(f"Версии одного документа: {result.evidence}.")
    else:
        lines.append(f"⚠️ Не похоже на версии одного документа ({result.evidence}) — сравнение по вашей просьбе.")
    lines += [
        "",
        f"- Изменено: **{result.count(sc.STATUS_CHANGED)}** — значения различаются, в обеих редакциях прочитаны уверенно.",
        f"- ⚠️ Проверить: **{result.count(sc.STATUS_CHECK)}** — различия, где значение прочитано неуверенно "
        "(правка от руки, штамп, плохой скан): сверьте со сканами.",
        f"- Добавлено: {result.count(sc.STATUS_ADDED)}, удалено: {result.count(sc.STATUS_REMOVED)}, "
        f"без изменений: {result.count(sc.STATUS_SAME)}.",
    ]
    for status, side in ((sc.STATUS_NEW_SHEET, "новой"), (sc.STATUS_OLD_SHEET, "старой")):
        pages = result.pages(status)
        if pages:
            lines.append(
                f"- Листы только в {side} редакции: стр. {_page_ranges(pages)} ({result.count(status)} позиций) — "
                "в другой нет ни одной общей позиции (например, изменение выпущено только на заменяемые листы)."
            )
    changed = [r for r in result.rows if r.status == sc.STATUS_CHANGED]
    if changed:
        lines.append("")
        rows = [[str(r.new.page), r.new.cells[spec_registry.COL_NAME], "; ".join(r.changes)] for r in changed[:15]]
        lines.append(_md_table(["Стр.", "Наименование", "Что изменилось"], rows))
        if len(changed) > 15:
            lines.append(f"\n_…ещё {len(changed) - 15} изменений в файле_")
    lines.append("\nГотовый файл ниже — можно скачать. 👇")
    return "\n".join(lines)


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
