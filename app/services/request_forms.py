"""Автозаполнение шаблонов заявок на доступ к системам (Этап 7).

Пользователь свободным текстом в чате описывает данные сотрудника и просит
оформить заявку на конкретную систему. Ассистент определяет систему, через
LLM извлекает нужные поля, заполняет xlsx-шаблон и отдаёт готовый файл.

Шаблоны лежат в `app/form_templates/` и НЕ изменяются — заполняется копия в
памяти, файл на сервере не сохраняется (как и в Этапе 6).
"""

from __future__ import annotations

import base64
import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from app.schemas import AskAttachment


TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "form_templates"

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Глаголы-триггеры: пользователь именно просит ОФОРМИТЬ заявку, а не задаёт
# вопрос про систему. Без такого глагола интент не срабатывает — иначе обычный
# QA-вопрос «что за лаборатория ИИ?» ушёл бы в автозаполнение.
_FILL_VERBS = (
    "заполни",
    "заполнить",
    "оформи",
    "оформить",
    "сформируй",
    "сформировать",
    "сгенерируй",
    "сгенерировать",
    "создай заявк",
    "создать заявк",
    "состав",  # составь/составить
    "подготов",  # подготовь/подготовить
    "сделай заявк",
    "сделать заявк",
)

_MONTHS_GENITIVE = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}


@dataclass
class FormField:
    key: str
    title: str
    required: bool = False
    description: str = ""


@dataclass
class FormSpec:
    key: str
    title: str
    template: str  # имя файла в form_templates/
    keywords: tuple[str, ...]  # триггеры названия системы
    fields: list[FormField]
    filler: Callable[["FormSpec", dict], bytes]
    output_name: Callable[[dict], str]

    def fields_for_llm(self) -> list[dict]:
        return [
            {"key": f.key, "title": f.title, "description": f.description}
            for f in self.fields
        ]

    def missing_required(self, values: dict) -> list[FormField]:
        return [f for f in self.fields if f.required and not values.get(f.key)]


@dataclass
class FormResult:
    answer: str
    attachment: AskAttachment | None = None


# ----------------------------- утилиты заполнения ------------------------------


def _template_path(name: str) -> Path:
    path = TEMPLATES_DIR / name
    if not path.exists():
        raise RuntimeError(f"Шаблон заявки не найден на сервере: {name}")
    return path


def _join_person(parts: list[str | None], *, phone_prefix: str = "") -> str:
    """Склейка «ФИО, должность, телефон», пустые части пропускаются."""
    out: list[str] = []
    for idx, part in enumerate(parts):
        value = (part or "").strip()
        if not value:
            continue
        if idx == len(parts) - 1 and phone_prefix and out:
            out.append(f"{phone_prefix}{value}")
        else:
            out.append(value)
    return ", ".join(out)


_INLINE_CELL_RE = re.compile(
    r'<c\b([^>]*?)\bt="inlineStr"([^>]*)>\s*<is>(.*?)</is>\s*</c>',
    re.DOTALL,
)
_SHARED_CT = (
    '<Override PartName="/xl/sharedStrings.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
)
_SHARED_REL_TYPE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings"
)


def _to_shared_strings(data: bytes) -> bytes:
    """Конвертирует inline-строки openpyxl в таблицу общих строк.

    openpyxl (3.1.x) пишет строки как inlineStr и не создаёт sharedStrings.xml.
    Excel/Google Sheets это читают, а Apple Numbers — нет. Переписываем книгу
    с `xl/sharedStrings.xml`, чтобы файл открывался везде. Поддерживаются
    простые строки `<is><t>…</t></is>` без rich-text (в наших шаблонах так и есть)
    — на всякий случай при наличии runs (`<is><r>`) конвертацию пропускаем.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        names = zin.namelist()
        members = {name: zin.read(name) for name in names}

    if "xl/sharedStrings.xml" in members:
        return data  # уже shared — ничего не делаем

    unique: list[str] = []
    index_by_inner: dict[str, int] = {}
    total = 0

    def _replace(match: re.Match) -> str:
        nonlocal total
        attrs = (match.group(1) + match.group(2)).rstrip()
        inner = match.group(3)
        idx = index_by_inner.get(inner)
        if idx is None:
            idx = len(unique)
            index_by_inner[inner] = idx
            unique.append(inner)
        total += 1
        return f'<c{attrs} t="s"><v>{idx}</v></c>'

    sheet_names = [n for n in names if n.startswith("xl/worksheets/") and n.endswith(".xml")]
    touched = False
    for name in sheet_names:
        xml = members[name].decode("utf-8")
        if "<is><r>" in xml:  # rich-text inline — не трогаем, отдаём как есть
            return data
        new_xml, count = _INLINE_CELL_RE.subn(_replace, xml)
        if count:
            members[name] = new_xml.encode("utf-8")
            touched = True

    if not touched:
        return data

    sst = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{total}" uniqueCount="{len(unique)}">'
        + "".join(f"<si>{inner}</si>" for inner in unique)
        + "</sst>"
    )
    members["xl/sharedStrings.xml"] = sst.encode("utf-8")

    # Регистрируем часть в [Content_Types].xml.
    ct = members["[Content_Types].xml"].decode("utf-8")
    if "sharedStrings.xml" not in ct:
        ct = ct.replace("</Types>", _SHARED_CT + "</Types>")
        members["[Content_Types].xml"] = ct.encode("utf-8")

    # Добавляем relationship в xl/_rels/workbook.xml.rels.
    rels_name = "xl/_rels/workbook.xml.rels"
    rels = members[rels_name].decode("utf-8")
    existing_ids = [int(n) for n in re.findall(r'Id="rId(\d+)"', rels)]
    new_id = f"rId{(max(existing_ids) + 1) if existing_ids else 1}"
    rel = f'<Relationship Id="{new_id}" Type="{_SHARED_REL_TYPE}" Target="sharedStrings.xml"/>'
    rels = rels.replace("</Relationships>", rel + "</Relationships>")
    members[rels_name] = rels.encode("utf-8")

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in members:
            zout.writestr(name, members[name])
    return out.getvalue()


def _safe_filename(stem: str) -> str:
    cleaned = stem
    for ch in '/\\:*?"<>|':
        cleaned = cleaned.replace(ch, "-")
    cleaned = " ".join(cleaned.split())
    return cleaned or "Заявка"


# --------------------- заполнение «родного» Excel-пакета ----------------------
#
# openpyxl при load+save переписывает весь пакет в свой диалект (теряет
# printerSettings, worksheets/_rels, урезает namespaces). Excel и Google Sheets
# это читают, а Apple Numbers — нет. Поэтому для заявок патчим ИСХОДНЫЙ
# Excel-пакет напрямую: меняем только нужные ячейки и sharedStrings, всё
# остальное оставляем байт-в-байт. Так файл остаётся максимально «родным».


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _set_cell(sheet_xml: str, ref: str, body: str, *, t: str | None) -> str:
    """Заменяет содержимое ячейки `ref`, сохраняя её стиль (атрибут s)."""
    pat = re.compile(rf'<c r="{re.escape(ref)}"([^>]*?)(/>|>.*?</c>)', re.DOTALL)
    m = pat.search(sheet_xml)
    if not m:
        raise RuntimeError(f"Ячейка {ref} не найдена в шаблоне заявки")
    style_match = re.search(r'\ss="(\d+)"', m.group(1))
    style = f' s="{style_match.group(1)}"' if style_match else ""
    t_attr = f' t="{t}"' if t else ""
    repl = f'<c r="{ref}"{style}{t_attr}>{body}</c>'
    return sheet_xml[: m.start()] + repl + sheet_xml[m.end() :]


def _fill_template_raw(
    template_name: str,
    string_cells: dict[str, str],
    number_cells: dict[str, int],
) -> bytes:
    """Заполняет xlsx-шаблон, не пересобирая пакет (для совместимости с Numbers)."""
    with zipfile.ZipFile(_template_path(template_name)) as zin:
        order = zin.namelist()
        members = {name: zin.read(name) for name in order}

    sst_name = "xl/sharedStrings.xml"
    sst = members[sst_name].decode("utf-8")
    si_count = len(re.findall(r"<si[ >]", sst))

    # Добавляем строковые значения в конец таблицы общих строк.
    new_si: list[str] = []
    ref_to_idx: dict[str, int] = {}
    for ref, value in string_cells.items():
        ref_to_idx[ref] = si_count + len(new_si)
        new_si.append(f'<si><t xml:space="preserve">{_xml_escape(value)}</t></si>')

    if new_si:
        total = si_count + len(new_si)
        sst = sst.replace("</sst>", "".join(new_si) + "</sst>")
        sst = re.sub(r'\bcount="\d+"', f'count="{total}"', sst, count=1)
        sst = re.sub(r'uniqueCount="\d+"', f'uniqueCount="{total}"', sst, count=1)
        members[sst_name] = sst.encode("utf-8")

    sheet_name = "xl/worksheets/sheet1.xml"
    sheet = members[sheet_name].decode("utf-8")
    for ref, idx in ref_to_idx.items():
        sheet = _set_cell(sheet, ref, f"<v>{idx}</v>", t="s")
    for ref, num in number_cells.items():
        sheet = _set_cell(sheet, ref, f"<v>{num}</v>", t=None)
    members[sheet_name] = sheet.encode("utf-8")

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in order:  # сохраняем исходный порядок частей пакета
            zout.writestr(name, members[name])
    return out.getvalue()


# ------------------------------- Лаборатория ИИ -------------------------------


def _fill_lab_ai(spec: FormSpec, v: dict) -> bytes:
    """Заполняет Шаблон_Лаборатория.xlsx по логике VBA-макроса LaboratoryAI_3_1.

    Патчим исходный Excel-пакет напрямую (не через openpyxl save), чтобы файл
    открывался и в Apple Numbers.
    """
    division = (v.get("division") or "").strip()
    today = date.today()
    month = _MONTHS_GENITIVE[today.month]

    string_cells = {
        # Подразделение — в две ячейки (как в макросе: F15 и H47).
        "F15": division,
        "H47": division,
        # F16: ФИО, должность, телефон работника.
        "F16": _join_person(
            [v.get("employee_name"), v.get("employee_position"), v.get("employee_phone")]
        ),
        # F17: ФИО, должность, тел. руководителя.
        "F17": _join_person(
            [v.get("manager_name"), v.get("manager_position"), v.get("manager_phone")],
            phone_prefix="тел.",
        ),
        # F18: имя ключа ПКЗИ.
        "F18": (v.get("pkzi") or "").strip(),
        # Месяц прописью (период «с … по …»).
        "H31": month,
        "M31": month,
        # ФИО внизу: работник (B43) и руководитель (B50).
        "B43": (v.get("employee_name") or "").strip(),
        "B50": (v.get("manager_name") or "").strip(),
    }
    number_cells = {
        # Дата: текущая, период «с» текущего года «по» год+1 (как в макросе).
        "F31": today.day,
        "I31": today.year,
        "K31": today.day,
        "N31": today.year + 1,
    }
    return _fill_template_raw(spec.template, string_cells, number_cells)


def _lab_ai_output_name(v: dict) -> str:
    fio = (v.get("employee_name") or "").strip()
    return f"Заявка_{_safe_filename(fio)}.xlsx" if fio else "Заявка_Лаборатория.xlsx"


LAB_AI = FormSpec(
    key="lab_ai",
    title="Лаборатория ИИ",
    template="lab_ai.xlsx",
    keywords=(
        "лаборатори",
        "лаб ии",
        "лаб. ии",
        "стенд ии",
        "стенда ии",
        "стенд искусств",
        "искусственн",
    ),
    fields=[
        FormField("division", "Наименование структурного подразделения", required=True),
        FormField("employee_name", "ФИО работника", required=True),
        FormField("employee_position", "Должность работника"),
        FormField("employee_phone", "Телефон работника"),
        FormField("manager_name", "ФИО непосредственного руководителя", required=True),
        FormField("manager_position", "Должность руководителя"),
        FormField("manager_phone", "Телефон руководителя"),
        FormField("pkzi", "Имя ключа ПКЗИ", required=True),
    ],
    filler=_fill_lab_ai,
    output_name=_lab_ai_output_name,
)


# ------------------------------- реестр систем --------------------------------

FORMS: list[FormSpec] = [LAB_AI]


def detect_form_intent(message: str) -> FormSpec | None:
    """Определяет, просит ли пользователь оформить заявку на одну из систем.

    Требуется И глагол-триггер (оформи/заполни/…), И упоминание системы.
    Это намеренно консервативно, чтобы не перехватывать обычные вопросы.
    """
    text = message.lower()
    if not any(verb in text for verb in _FILL_VERBS):
        return None
    for spec in FORMS:
        if any(kw in text for kw in spec.keywords):
            return spec
    return None


def _missing_fields_answer(spec: FormSpec, missing: list[FormField], values: dict) -> str:
    have = [f for f in spec.fields if values.get(f.key)]
    lines = [f"Чтобы оформить заявку «{spec.title}», не хватает данных:"]
    lines += [f"- {f.title}" for f in missing]
    if have:
        lines.append("")
        lines.append("Уже распознано:")
        lines += [f"- {f.title}: {values[f.key]}" for f in have]
    lines.append("")
    lines.append("Допишите недостающее одним сообщением — и я сформирую файл.")
    return "\n".join(lines)


def handle_form_request(spec: FormSpec, message: str, llm) -> FormResult:
    """Извлекает поля из текста и заполняет шаблон. Ошибки LLM → RuntimeError."""
    values = llm.extract_fields(message, spec.fields_for_llm())

    missing = spec.missing_required(values)
    if missing:
        return FormResult(answer=_missing_fields_answer(spec, missing, values))

    content = _to_shared_strings(spec.filler(spec, values))
    filename = spec.output_name(values)
    attachment = AskAttachment(
        filename=filename,
        content_base64=base64.b64encode(content).decode("ascii"),
        mime=XLSX_MIME,
    )

    summary_lines = [f"✅ Заявка «{spec.title}» сформирована. Проверьте данные:"]
    for f in spec.fields:
        if values.get(f.key):
            summary_lines.append(f"- {f.title}: {values[f.key]}")
    summary_lines.append("")
    summary_lines.append(f"Файл «{filename}» приложен ниже — скачайте и при необходимости поправьте.")
    return FormResult(answer="\n".join(summary_lines), attachment=attachment)
