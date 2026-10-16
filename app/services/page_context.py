"""Что открыто на странице портала, с которой задан вопрос (кнопка ассистента, widget.js).

Панель присылает содержимое страницы: раздел, заголовок, вкладку, поля формы и фильтры,
выделенную строку, первые строки видимых таблиц, сообщения об ошибках и видимый текст. Здесь
оно ещё раз очищается (пароли, скрытые и служебные поля, похожие на ключи значения),
ограничивается по объёму и превращается в блок для модели. Это данные со страницы, а не
инструкции: модель получает их отдельным блоком с такой пометкой.

В журнал вопросов и в историю беседы попадает только короткая метка раздела, не текст страницы.
"""

from __future__ import annotations

import re
from datetime import date
from itertools import combinations

from pydantic import BaseModel, Field

from app.services.grounding import _STOPWORDS

MAX_CHARS = 3500  # весь блок страницы для модели
MAX_LABEL = 120

# Разделы портала по началу адреса (подписи — как на плитках портала).
SECTIONS: tuple[tuple[str, str], ...] = (
    ("/metrology/", "Метрология"),
    ("/siz/", "СИЗ"),
    ("/tkzaii/", "ТКзаИИ"),
    ("/protocol/", "Протокол"),
    ("/development-plan/", "План развития"),
    ("/metrics/", "Метрики"),
    ("/tasks/", "Панель задач"),
    ("/assistant/tools/", "Конструктор и чертежи"),
    ("/account/", "Учётная запись"),
    ("/admin/", "Администрирование портала"),
)
HOME = "Главная портала"

# Поля, значения которых не передаются никогда (страница их тоже не шлёт, это вторая линия).
_SECRET_LABEL = re.compile(r"парол|password|passwd|token|токен|csrf|secret|секрет|ключ|api[-_ ]?key|otp|pin", re.I)
# Длинные строки из букв, цифр и знаков base64 — похожи на ключи и пропуска.
_SECRET_VALUE = re.compile(r"[A-Za-z0-9+/=_.-]{32,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


class PageField(BaseModel):
    label: str = Field("", max_length=200)
    value: str = Field("", max_length=500)


class PageTable(BaseModel):
    caption: str = Field("", max_length=200)
    headers: list[str] = Field(default_factory=list, max_length=40)
    rows: list[list[str]] = Field(default_factory=list, max_length=20)
    total_rows: int | None = Field(None, ge=0)


class PageContext(BaseModel):
    path: str = Field("", max_length=300)
    title: str = Field("", max_length=300)
    heading: str = Field("", max_length=300)
    tab: str = Field("", max_length=200)
    fields: list[PageField] = Field(default_factory=list, max_length=60)
    selected: str = Field("", max_length=1000)
    tables: list[PageTable] = Field(default_factory=list, max_length=5)
    errors: list[str] = Field(default_factory=list, max_length=10)
    text: str = Field("", max_length=6000)


def _clean(value: str, limit: int) -> str:
    value = _CONTROL.sub(" ", value or "")
    value = _SPACES.sub(" ", value).strip()
    value = _SECRET_VALUE.sub("…", value)
    return value[:limit]


def section_for(path: str) -> str:
    path = (path or "/").split("?", 1)[0]
    if path in ("", "/"):
        return HOME
    for prefix, name in SECTIONS:
        if path == prefix.rstrip("/") or path.startswith(prefix):
            return name
    return "Портал"


def label(ctx: PageContext) -> str:
    """Короткая метка для журнала и истории: «Метрология → Реестр средств измерений»."""
    section = section_for(ctx.path)
    page = _clean(ctx.heading or ctx.title, 80)
    if page and page.lower() != section.lower():
        return f"{section} → {page}"[:MAX_LABEL]
    return section[:MAX_LABEL]


def sanitize(ctx: PageContext) -> PageContext:
    return PageContext(
        path=_clean(ctx.path.split("?", 1)[0], 300),
        title=_clean(ctx.title, 200),
        heading=_clean(ctx.heading, 200),
        tab=_clean(ctx.tab, 100),
        fields=[
            PageField(label=_clean(f.label, 100), value=_clean(f.value, 200))
            for f in ctx.fields
            if f.label.strip() and not _SECRET_LABEL.search(f.label)
        ],
        selected=_clean(ctx.selected, 500),
        tables=[
            PageTable(
                caption=_clean(t.caption, 100),
                headers=[_clean(h, 60) for h in t.headers],
                rows=[[_clean(c, 80) for c in row[:40]] for row in t.rows],
                total_rows=t.total_rows,
            )
            for t in ctx.tables
        ],
        errors=[e for e in (_clean(e, 300) for e in ctx.errors) if e],
        text=_BLANK_LINES.sub("\n", _clean(ctx.text, 6000)),
    )


def render(ctx: PageContext, limit: int = MAX_CHARS) -> str:
    """Блок «что на странице» для модели. Сначала самое важное — оно не обрезается."""
    ctx = sanitize(ctx)
    lines = [f"Раздел портала: {section_for(ctx.path)}"]
    computed = facts(ctx)
    if ctx.heading or ctx.title:
        lines.append(f"Страница: {ctx.heading or ctx.title}")
    if ctx.tab:
        lines.append(f"Активная вкладка: {ctx.tab}")
    if ctx.errors:
        lines.append("Сообщения об ошибках на странице:")
        lines += [f"- {e}" for e in ctx.errors]
    if ctx.selected:
        lines.append(f"Выделенная строка: {ctx.selected}")
    if ctx.fields:
        lines.append("Поля и фильтры (подпись: значение):")
        lines += [f"- {f.label}: {_shown(f.value) or '(пусто)'}" for f in ctx.fields]
    if computed:
        lines.append("Посчитано ассистентом по значениям на странице (точно):")
        lines += [f"- {fact}" for fact in computed]
    for table in ctx.tables:
        name = f"«{table.caption}»" if table.caption else "на странице"
        shown = len(table.rows)
        total = f", всего строк: {table.total_rows}" if table.total_rows is not None else ""
        lines.append(f"Таблица {name} (показано строк: {shown}{total}):")
        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
        lines += ["| " + " | ".join(row) + " |" for row in table.rows]
    head = "\n".join(lines)
    if len(head) > limit:
        return head[: limit - 1] + "…"
    if ctx.text and len(head) + 30 < limit:
        rest = limit - len(head) - len("\nТекст страницы:\n")
        text = ctx.text if len(ctx.text) <= rest else ctx.text[: rest - 1] + "…"
        head += "\nТекст страницы:\n" + text
    return head


def table_rows(ctx: PageContext) -> int:
    return sum(len(t.rows) for t in ctx.tables)


# Вопрос о том, что открыто на экране, — отвечать по странице, а не искать в базе знаний.
_SCREEN = re.compile(
    r"(на\s+экран|экран[еау]?\b|эт(а|у|ой|от|ом|о)\s+(страниц|раздел|таблиц|форм|пол[еяю]|кнопк|вкладк|фильтр|ошибк|окн|отчёт|отчет|дашборд)"
    r"|на\s+(этой\s+)?странице|здесь|тут\b|где\s+я\b|что\s+я\s+вижу|что\s+(показано|представлено|отображ|открыт|видно)"
    r"|что\s+(это\s+)?(за\s+)?(страниц|раздел|таблиц|форм|ошибк)|ошибк[аиу]\s+на\b)",
    re.I,
)


def is_screen_question(question: str) -> bool:
    return bool(_SCREEN.search(question or ""))


# Вопрос о данных, которые пользователь ввёл или выбрал на странице.
_INPUT = re.compile(
    r"\bя\s+(указал|ввел|ввёл|заполнил|выбрал|поставил|написал|внес|внёс|отметил)а?\b"
    r"|\bмо(и|ё|е|й|я|ем|ей|их)\s+(данн|значен|дат|пол|запис|ввод|цифр|числ)"
    r"|\b(указанн|введенн|введённ|заполненн|выбранн|внесенн|внесённ)\w*",
    re.I,
)


def _stems4(text: str) -> set[str]:
    words = (w.lower().replace("ё", "е") for w in re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", text or ""))
    return {w[:4] for w in words if w not in _STOPWORDS}


def mentions_field(question: str, ctx: PageContext) -> bool:
    """В вопросе названо поле со страницы — хотя бы два слова его подписи («между датами
    поверки» → «Дата следующей поверки»). Одного слова мало: «статусы заявки в ЕКТП» —
    вопрос к базе знаний, даже если на странице есть фильтр «Статус»."""
    asked = _stems4(question)
    for field in ctx.fields:
        stems = _stems4(field.label)
        common = stems & asked
        if len(common) >= 2:
            return True
    return False


def is_page_question(question: str, ctx: PageContext) -> bool:
    """Отвечать по странице: вопрос об экране, о введённых данных или о поле со страницы."""
    return is_screen_question(question) or bool(_INPUT.search(question or "")) or mentions_field(question, ctx)


# ---------- расчёты по значениям на странице ----------
# Маленькая модель плохо считает даты и суммы: считаем сами и отдаём ей готовые факты.

_DATE_FORMATS = (
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"), (0, 1, 2)),
    (re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$"), (2, 1, 0)),
)
# Пары «раньше — позже» по подписи поля: текущая/следующая, начало/окончание, выдача/списание.
_EARLY = re.compile(r"текущ|начал|выдач|последн|прошл|\bот\b|\bс\b", re.I)
_LATE = re.compile(r"следующ|окончан|истеч|списан|завершен|план|\bдо\b|\bпо\b", re.I)
_NUMBER = re.compile(r"^-?\d+(?:[.,]\d+)?$")
MAX_DATE_FIELDS = 6


def parse_date(value: str) -> date | None:
    value = (value or "").strip()
    for pattern, order in _DATE_FORMATS:
        m = pattern.match(value)
        if m:
            parts = [int(g) for g in m.groups()]
            try:
                return date(parts[order[0]], parts[order[1]], parts[order[2]])
            except ValueError:
                return None
    return None


def _fmt(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def _shown(value: str) -> str:
    """Значение поля для модели: даты — в одном виде (поле даты отдаёт 2026-10-01)."""
    d = parse_date(value)
    return _fmt(d) if d else value


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    for day in (d.day, 30, 29, 28):
        try:
            return date(year, month, day)
        except ValueError:
            continue
    raise ValueError(d)


def span(a: date, b: date) -> str:
    """Промежуток между датами словами: «1 мес. 5 дн.» (без знака)."""
    a, b = min(a, b), max(a, b)
    months = (b.year - a.year) * 12 + (b.month - a.month)
    if _add_months(a, months) > b:
        months -= 1
    days = (b - _add_months(a, months)).days
    parts = [f"{months} мес." if months else "", f"{days} дн." if days else ""]
    return " ".join(p for p in parts if p) or "0 дн."


def _number(value: str) -> float | None:
    value = value.replace("\u00a0", "").replace(" ", "")
    return float(value.replace(",", ".")) if _NUMBER.match(value) else None


def facts(ctx: PageContext, today: date | None = None) -> list[str]:
    today = today or date.today()
    out: list[str] = []
    dates = [(f.label, d) for f in ctx.fields if (d := parse_date(f.value))][:MAX_DATE_FIELDS]
    for label, d in dates:
        delta = (d - today).days
        when = "сегодня" if delta == 0 else f"через {span(today, d)}" if delta > 0 else f"{span(d, today)} назад"
        out.append(f"«{label}» = {_fmt(d)}; относительно сегодняшней даты — {when}")
    for (la, a), (lb, b) in combinations(dates, 2):
        if a == b:
            out.append(f"«{la}» и «{lb}» совпадают")
            continue
        out.append(f"Промежуток между «{la}» и «{lb}»: {span(a, b)}")
        earlier, later = (la, lb) if a < b else (lb, la)
        # подпись говорит, что поле должно быть позже, а оно раньше
        if _LATE.search(earlier) and _EARLY.search(later) and not _LATE.search(later):
            out.append(f"Ошибка порядка дат: «{earlier}» раньше «{later}», а должна быть позже")
        else:
            out.append(f"Порядок дат: «{earlier}» раньше «{later}»")
    for table in ctx.tables:
        name = f"«{table.caption}»" if table.caption else "на странице"
        for i, header in enumerate(table.headers):
            values = [row[i] for row in table.rows if i < len(row) and row[i].strip()]
            numbers = [n for n in (_number(v) for v in values) if n is not None]
            if len(numbers) >= 2 and len(numbers) == len(values):
                total = sum(numbers)
                shown = f"{total:g}".replace(".", ",")
                out.append(f"Таблица {name}: сумма по колонке «{header}» в показанных строках ({len(numbers)}) = {shown}")
    if out:
        out.insert(0, f"Сегодня: {_fmt(today)}")
    return out
