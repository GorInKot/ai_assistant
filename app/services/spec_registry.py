"""Реестр позиций спецификаций (форма ГОСТ 21.110) по сканам PDF.

Модель здесь не участвует: система знает бланк. На странице ищется таблица из
9 граф по пропорциям формы, линии сетки стираются, каждая графа распознаётся
Tesseract со своими настройками (цифры — как цифры, единицы — по-русски), затем
строки собираются в позиции. Каждое сомнительное значение помечается
«Проверить», а не принимается молча.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

DPI = 300
MM = DPI / 25.4  # пикселей в миллиметре
COLOR_CAST_SHARE = 0.25  # «цветными» оказалось больше четверти чернил листа — это оттенок скана

COLUMNS = (
    "Позиция",
    "Наименование и техническая характеристика",
    "Тип, марка, обозначение документа",
    "Код",
    "Завод-изготовитель",
    "Ед. изм.",
    "Кол-во",
    "Масса ед., кг",
    "Примечание",
)
COL_POS, COL_NAME, COL_TYPE, COL_CODE, COL_MAKER, COL_UNIT, COL_QTY, COL_MASS, COL_NOTE = range(9)

# Ширины граф формы 1 ГОСТ 21.110-2013, мм. Сканы бывают в другом масштабе, поэтому
# сравниваются доли, а не миллиметры.
FORM_WIDTHS_MM = (20, 130, 60, 35, 45, 20, 20, 25, 40)
FORM_SHARES = np.cumsum((0,) + FORM_WIDTHS_MM) / sum(FORM_WIDTHS_MM)  # 10 границ, 0..1
LINE_TOLERANCE = 0.02  # допуск положения границы, доля ширины таблицы (~8 мм на A3): у бланков A4 ширины граф свои
MIN_MATCHED_INNER = 6  # из 8 внутренних границ должны найтись хотя бы 6
MAX_EXTRA_LINES = 3  # столько длинных лишних вертикалей внутри — другая форма (ведомость и т.п.)


class SpecFormNotFound(ValueError):
    """На странице нет таблицы формы спецификации."""


@dataclass
class Grid:
    """Таблица спецификации на странице (координаты в пикселях выровненного изображения)."""

    xs: list[int]  # 10 границ граф слева направо
    table_top: int  # верх таблицы (над шапкой)
    top: int  # верх тела таблицы (под шапкой и строкой номеров граф)
    bottom: int  # низ тела таблицы (верх основной надписи)
    matched: int  # сколько внутренних границ найдено линиями (остальные — по пропорции)
    rows: list[int] = field(default_factory=list)  # горизонтальные линейки внутри тела


def binarize(gray: np.ndarray) -> np.ndarray:
    """Чернила → 255, фон → 0. Адаптивный порог переживает неровную яркость скана."""
    import cv2

    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)


def line_masks(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Маски вертикальных и горизонтальных линий: всё, что тянется дальше ~12 мм."""
    import cv2

    h, w = ink.shape
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(12 * MM))))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(12 * MM), 1)))
    # Сканы с разрывами линий: сшиваем короткие пропуски
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, int(3 * MM))))
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * MM), 3)))
    return vert, horiz


def skew_angle(horiz: np.ndarray) -> float:
    """Перекос листа в градусах по длинным горизонтальным линиям (0, если линий нет)."""
    import cv2

    count, _labels, stats, _ = cv2.connectedComponentsWithStats(horiz)
    angles = []
    for i in range(1, count):
        x, y, w, h, _area = stats[i]
        if w < 80 * MM:
            continue
        ys, xs = np.nonzero(_labels[y:y + h, x:x + w] == i)
        if len(xs) < 50:
            continue
        slope = np.polyfit(xs, ys, 1)[0]
        angles.append(np.degrees(np.arctan(slope)))
    return float(np.median(angles)) if angles else 0.0


def rotate(gray: np.ndarray, angle: float, border: int = 255) -> np.ndarray:
    """Повернуть на малый угол вокруг центра (по умолчанию белый фон)."""
    import cv2

    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


def _runs(mask: np.ndarray, gap: int = 3) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


@dataclass
class VLine:
    x: int
    y0: int  # самый длинный отрезок линии
    y1: int
    start: int  # самое верхнее начало линии (отрезки от 10 мм): линия бывает прервана

    @property
    def length(self) -> int:
        return self.y1 - self.y0


def vertical_lines(vert: np.ndarray, min_len: float) -> list[VLine]:
    """Вертикальные линии длиннее min_len пикселей: x и протяжённость по y."""
    profile = (vert > 0).sum(axis=0)
    lines = []
    for a, b in _runs(profile > min_len * 0.5):
        band = vert[:, a:b + 1].max(axis=1) > 0
        all_spans = [s for s in _runs(band, gap=int(4 * MM)) if s[1] - s[0] >= 10 * MM]
        spans = [s for s in all_spans if s[1] - s[0] >= min_len]
        if spans:
            y0, y1 = max(spans, key=lambda s: s[1] - s[0])
            lines.append(VLine(x=(a + b) // 2, y0=y0, y1=y1, start=min(s[0] for s in all_spans)))
    return lines


def match_form(lines: list[VLine], page_width: int) -> tuple[list[int], int, list[VLine]] | None:
    """Найти среди линий 10 границ граф формы по пропорциям. Возвращает (xs, найдено, линии).

    Из подходящих вариантов берётся тот, где найдено больше границ; при равенстве — где
    отклонение от пропорций меньше и внешние линии длиннее (линии боковой графы
    «Инв. № подл.» у левого края короткие и иначе могли бы сойти за границу таблицы)."""
    xs = [ln.x for ln in lines]
    best, best_score = None, None
    for i, left in enumerate(xs):
        for j in range(len(xs) - 1, i, -1):
            right = xs[j]
            width = right - left
            if width < page_width * 0.45:  # таблица занимает большую часть листа
                break
            bounds, used, matched, deviation = [left], [lines[i]], 0, 0.0
            for share in FORM_SHARES[1:-1]:
                expect = left + share * width
                near = min(lines, key=lambda ln: abs(ln.x - expect))
                if abs(near.x - expect) <= LINE_TOLERANCE * width:
                    bounds.append(near.x)
                    used.append(near)
                    matched += 1
                    deviation += abs(near.x - expect) / width
                else:
                    bounds.append(int(round(expect)))
            bounds.append(right)
            used.append(lines[j])
            if matched < MIN_MATCHED_INNER:
                continue
            outer = min(lines[i].y1 - lines[i].start, lines[j].y1 - lines[j].start)
            score = (matched, round(-deviation, 3), outer)
            if best_score is None or score > best_score:
                best, best_score = (bounds, matched, used), score
    return best


def find_grid(gray: np.ndarray) -> tuple[Grid, np.ndarray]:
    """Найти таблицу спецификации на выровненной странице. Возвращает сетку и маску линий."""
    ink = binarize(gray)
    vert, horiz = line_masks(ink)
    h, w = gray.shape
    lines = vertical_lines(vert, min_len=0.2 * h)
    found = match_form(lines, w)
    if found is None:
        raise SpecFormNotFound("на странице нет таблицы формы спецификации (9 граф)")
    xs, matched, used = found
    # Верх таблицы — по внешним границам: внутренние графы прерываются строками раздела
    # во всю ширину («2. Трубопроводы…»). Низ — по внутренним графам: они упираются в
    # основную надпись, а внешние линии могут идти до рамки листа.
    top = min(used[0].start, used[-1].start)
    bottom = int(np.median([ln.y1 for ln in used[1:-1]]))
    left, right = xs[0], xs[-1]
    # У спецификации внутри таблицы ровно 8 линий граф. У ведомости оборудования (18 граф)
    # и других форм среди лишних линий находятся «подходящие» по пропорциям — отсекаем.
    tol = 0.01 * (right - left)
    extra = [ln for ln in lines if left + tol < ln.x < right - tol
             and min(abs(ln.x - b) for b in xs) > tol and ln.length >= 0.5 * (bottom - top)]
    if len(extra) >= MAX_EXTRA_LINES:
        raise SpecFormNotFound(f"в таблице {len(xs) - 1 + len(extra)} граф — это не форма спецификации")
    rows_profile = (horiz[:, left:right] > 0).mean(axis=1)
    rules = [(a + b) // 2 for a, b in _runs(rows_profile > 0.6) if top - 5 * MM <= (a + b) // 2 <= bottom + 5 * MM]
    body_top = _body_top(rules, top)
    dashed, dashed_mask = dashed_rules(ink, left, right, body_top, bottom)
    rows = sorted(set(y for y in rules if body_top < y < bottom - 2 * MM) | set(dashed))
    rows = [y for i, y in enumerate(rows) if i == 0 or y - rows[i - 1] > 1.5 * MM]
    grid = Grid(xs=xs, table_top=top, top=body_top, bottom=bottom, matched=matched, rows=rows)
    return grid, np.maximum(np.maximum(vert, horiz), dashed_mask)


def dashed_rules(ink: np.ndarray, left: int, right: int, top: int, bottom: int) -> tuple[list[int], np.ndarray]:
    """Пунктирные разделители строк (y) и их маска.

    Штрихи склеиваются короткой горизонтальной «перемычкой», после чего пунктир становится
    линией. Строку текста перемычка тоже склеит, но она толще: линии отбираются по толщине
    (до 1,6 мм; склеенная строка текста — от 2,3 мм) и длине (больше половины ширины таблицы)."""
    import cv2

    mask = np.zeros_like(ink)
    area = ink[top:bottom, left:right]
    if area.size == 0:
        return [], mask
    joined = cv2.morphologyEx(area, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(2.5 * MM), 1)))
    long = cv2.morphologyEx(joined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(12 * MM), 1)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(long)
    found = []
    for i in range(1, count):
        x, y, w, h, _area = stats[i]
        if h <= 1.6 * MM and w >= 0.5 * (right - left):
            found.append(top + y + h // 2)
            mask[top + y:top + y + h, left + x:left + x + w] = np.where(labels[y:y + h, x:x + w] == i, 255, 0)
    return sorted(found), mask


# Слова шапки формы спецификации (с учётом переносов: «Коли-чество», «измере-ния»).
HEADER_WORDS = ("наимен", "позиц", "марк", "изготов", "един", "коли", "масс", "примеч")
MIN_HEADER_WORDS = 2  # у листа регистрации изменений из этих слов нет ни одного


def header_matches(img: np.ndarray, grid: Grid) -> bool:
    """Шапка таблицы похожа на шапку спецификации (отсекает листы регистрации изменений и т.п.)."""
    import pytesseract
    from PIL import Image

    y0, y1 = grid.table_top, max(grid.top, grid.table_top + int(50 * MM))
    area = img[y0:y1, grid.xs[0]:grid.xs[-1]]
    vert, horiz = line_masks(binarize(area))
    crop = Image.fromarray(erase_lines(area, np.maximum(vert, horiz)))  # линии граф мешают OCR
    try:
        text = pytesseract.image_to_string(crop, lang="rus", config="--psm 6").lower()
    except Exception:  # noqa: BLE001 — без OCR проверить нельзя: доверяемся пропорциям граф
        return True
    return sum(word in text for word in HEADER_WORDS) >= MIN_HEADER_WORDS


MAX_FORM_AREA_MM2 = 841 * 594  # лист больше A1 — чертёж, а не спецификация; при 300 dpi это >70 Мп


def too_large_for_form(page) -> bool:
    """Огромный лист (чертёж в длину) не рендерим ради поиска формы: на 8 ГБ памяти опасно."""
    return (page.rect.width / 72 * 25.4) * (page.rect.height / 72 * 25.4) > MAX_FORM_AREA_MM2


def render_page(page) -> tuple[np.ndarray, np.ndarray]:
    """Страница PDF → (полутоновое изображение 300 dpi, маска цветных чернил).

    Цветные чернила на цветном скане — это ручки и штампы: правки количеств, пометки
    «изм.», согласования. Строки с ними помечаются «Проверить»."""
    import cv2

    pix = page.get_pixmap(dpi=DPI)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colored = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] < 235)).astype(np.uint8) * 255
    # Цветная кайма по краям чёрных букв (сканер сдвигает каналы) тоньше штриха ручки:
    # 1–2 пикселя против 4–6 — убирается «открытием» 5×5, пометки остаются.
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Цветной оттенок всего скана (так печатают иные МФУ) — не пометки: ручка и штамп
    # занимают малую долю чернил листа, а оттенок — почти все.
    ink = gray < 160
    if ink.any() and (colored[ink] > 0).mean() > COLOR_CAST_SHARE:
        colored[:] = 0
    return gray, colored


def osd_rotation(gray: np.ndarray) -> int:
    """На сколько градусов (0/90/180/270) повернуть страницу, по мнению Tesseract OSD."""
    import pytesseract
    from PIL import Image

    small = Image.fromarray(gray).reduce(2)
    try:
        osd = pytesseract.image_to_osd(small, config="--psm 0 -c min_characters_to_try=20")
        return int(re.search(r"Rotate:\s*(\d+)", osd).group(1)) % 360
    except Exception:  # noqa: BLE001 — мало текста или OSD недоступен: попробуем все повороты
        return 0


def _rot90(gray: np.ndarray, angle: int) -> np.ndarray:
    """Повернуть по часовой стрелке на angle (кратно 90)."""
    return np.rot90(gray, k=-(angle // 90) % 4).copy()


@dataclass
class Located:
    """Страница, повёрнутая и выровненная, с найденной таблицей."""

    img: np.ndarray  # полутоновое изображение
    colored: np.ndarray  # маска цветных чернил в тех же координатах
    grid: Grid
    lines: np.ndarray  # маска линий сетки
    rotation: int  # поворот кратно 90°
    skew: float  # выровненный перекос, градусы


def locate_form(gray: np.ndarray, colored: np.ndarray | None = None) -> Located:
    """Повернуть страницу как надо, выровнять перекос и найти таблицу.

    Сначала поворот по OSD, при неудаче — остальные."""
    if colored is None:
        colored = np.zeros_like(gray)
    first = osd_rotation(gray)
    for angle in dict.fromkeys((first, 0, 90, 270, 180)):
        img, col = _rot90(gray, angle), _rot90(colored, angle)
        _vert, horiz = line_masks(binarize(img))
        skew = skew_angle(horiz)
        if abs(skew) > 0.05:
            img, col = rotate(img, skew), rotate(col, skew, border=0)
        try:
            grid, lines = find_grid(img)
        except SpecFormNotFound:
            continue
        if not header_matches(img, grid):
            continue
        return Located(img=img, colored=col, grid=grid, lines=lines, rotation=angle, skew=skew)
    raise SpecFormNotFound("на странице нет таблицы формы спецификации (9 граф)")


def _body_top(rules: list[int], top: int) -> int:
    """Верх тела таблицы: под строкой номеров граф «1…9».

    Над шапкой бывает строка заголовка («СПЕЦИФИКАЦИЯ ОБОРУДОВАНИЯ»), поэтому ищем
    характерную пару: высокая шапка (12+ мм) и сразу под ней узкая строка номеров (3–11 мм).
    Без строки номеров — низ самой высокой полосы в верхних 70 мм таблицы."""
    marks = [top] + [y for y in rules if y > top + 2 * MM]
    for prev, header, numbers in zip(marks, marks[1:], marks[2:]):
        if header - prev >= 12 * MM and 3 * MM <= numbers - header <= 11 * MM:
            return numbers
    bands = [(b - a, b) for a, b in zip(marks, marks[1:]) if b - top <= 70 * MM]
    if bands:
        return max(bands)[1]
    return top + int(25 * MM)


# ------------------------------ распознавание -----------------------------

# Своё распознавание для коротких граф: в количестве и массе только цифры, единицы —
# по-русски (иначе «шт.» читается как «LUT.», а «м» теряется). (язык, настройки, мин. уверенность)
_TEXT_OCR = ("rus+eng", "--psm 4", 30)
_DIGITS = "--psm 6 -c tessedit_char_whitelist=0123456789,.*-"
COLUMN_OCR = {
    COL_QTY: ("eng", _DIGITS, 0),
    COL_MASS: ("eng", _DIGITS, 0),
}
CELL_PAD = 4  # отступ от линии графы, пикселей
LOW_CONFIDENCE = 55  # уверенность Tesseract в количестве, массе, единице ниже — «Проверить»
POOR_NAME_CONFIDENCE = 30  # наименование: обозначения из букв и цифр (марки арматуры) читаются верно и при 40–50


@dataclass
class Fragment:
    """Строка текста внутри одной графы."""

    col: int
    y: float  # центр по вертикали
    height: float
    text: str  # без зачёркнутых слов
    conf: float  # минимальная уверенность по словам
    x0: float = 0.0  # левый и правый край текста на странице
    x1: float = 0.0
    struck: list[str] = field(default_factory=list)  # зачёркнутые слова (правка по документу)
    image: np.ndarray | None = None  # чернила единицы измерения (для сверки с образцами)


@dataclass
class TableLine:
    """Строка таблицы: текст по 9 графам."""

    y: float
    cells: list[str]
    conf: list[float]  # минимальная уверенность по графе (100 — пусто)
    colored: bool = False  # рядом цветные чернила (ручка, штамп)
    struck: dict[int, list[str]] = field(default_factory=dict)  # графа → зачёркнутые слова
    header_like: bool = False  # оформлена как заголовок раздела: подчёркнута, по центру, с номером
    name_x: tuple[float, float] = (0.0, 0.0)  # края текста в графе «Наименование»
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)  # (графа, x0, y0, x1, y1)
    foreign_ink: bool = False  # в графах чисел чернила, не объяснённые распознанным текстом

    def text(self, col: int) -> str:
        return self.cells[col]


def erase_lines(img: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Стереть линии сетки белым: иначе Tesseract читает их как буквы («ПО ОО»)."""
    import cv2

    clean = img.copy()
    clean[cv2.dilate(lines, np.ones((5, 5), np.uint8)) > 0] = 255
    return clean


STRIKE_COLUMNS = (COL_NAME, COL_TYPE, COL_QTY, COL_MASS)  # где зачёркивание меняет смысл


def is_struck(ink: np.ndarray) -> bool:
    """Слово перечёркнуто: в средней полосе высоты есть сплошной штрих почти во всю ширину.

    Буквы («Н», «е», «ш») дают горизонтали короче слова; подчёркивание лежит ниже середины.
    Одиночные знаки не проверяются: у «-» и «7» штрих во всю ширину свой."""
    h, w = ink.shape
    if h < 8 or w < 1.3 * h:
        return False
    for row in ink[int(0.3 * h):int(0.75 * h) + 1]:
        runs = _runs(row, gap=2)
        if runs and max(b - a + 1 for a, b in runs) >= 0.85 * w:
            return True
    return False


AGREED_CONFIDENCE = 90.0  # значение подтверждено вторым прочтением или образцом


def _reread_number(box: np.ndarray) -> str:
    """Прочитать число ещё раз: одной строкой, увеличенным вдвое. Совпало с первым — верим."""
    import pytesseract
    from PIL import Image

    img = Image.fromarray(np.pad(box, 12, constant_values=255))
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789,.*-"
    try:
        return pytesseract.image_to_string(img, lang="eng", config=config).strip()
    except Exception:  # noqa: BLE001 — нет второго прочтения — остаётся неуверенным
        return ""


def ocr_column(clean: np.ndarray, grid: Grid, col: int, original: np.ndarray | None = None) -> list[Fragment]:
    """Распознать одну графу тела таблицы: строки текста с положением и уверенностью.

    Зачёркнутые слова (по исходному скану: на очищенном длинные зачёркивания стёрты вместе
    с линиями) в текст не попадают, а собираются отдельно — это правка по документу."""
    import pytesseract
    from PIL import Image

    x0, x1 = grid.xs[col] + CELL_PAD, grid.xs[col + 1] - CELL_PAD
    if x1 - x0 < 10:
        return []
    crop = Image.fromarray(clean[grid.top:grid.bottom, x0:x1])
    lang, config, min_conf = COLUMN_OCR.get(col, _TEXT_OCR)
    data = pytesseract.image_to_data(crop, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    check_strike = original is not None and col in STRIKE_COLUMNS
    words: dict[tuple, list] = {}
    for i, raw in enumerate(data["text"]):
        text, conf = raw.strip(), float(data["conf"][i])
        if not text or conf < min_conf:
            continue
        left, top, width, height = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if col in (COL_QTY, COL_MASS) and conf < LOW_CONFIDENCE:
            box = clean[grid.top + top:grid.top + top + height, x0 + left:x0 + left + width]
            if box.size and _reread_number(box) == text:
                conf = AGREED_CONFIDENCE  # два прочтения сошлись
        struck = False
        if check_strike:
            box = original[grid.top + top:grid.top + top + height, x0 + left:x0 + left + width]
            struck = box.size > 0 and is_struck(box < 140)
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        words.setdefault(key, []).append((left, top, height, text, conf, width, struck))
    fragments = []
    for ws in words.values():
        ws.sort()
        ys = [w[1] + w[2] / 2 for w in ws]
        fragments.append(Fragment(
            col=col,
            y=grid.top + sum(ys) / len(ys),
            height=float(np.median([w[2] for w in ws])),
            text=" ".join(w[3] for w in ws if not w[6]),
            conf=min(w[4] for w in ws),
            x0=x0 + ws[0][0],
            x1=x0 + ws[-1][0] + ws[-1][5],
            struck=[w[3] for w in ws if w[6]],
        ))
    return fragments


def assemble_lines(fragments: list[Fragment], grid: Grid) -> list[TableLine]:
    """Собрать фрагменты граф в строки таблицы по вертикали.

    Фрагменты ближе половины высоты строки — одна строка. Если в таблице есть линейки,
    фрагменты между одними и теми же линейками тоже одна строка (две строки мелкого
    текста в одной клетке)."""
    if not fragments:
        return []
    height = float(np.median([f.height for f in fragments])) or 30.0
    band_of = (lambda y: int(np.searchsorted(grid.rows, y))) if len(grid.rows) >= 3 else (lambda y: None)
    groups: list[list[Fragment]] = []
    for frag in sorted(fragments, key=lambda f: f.y):
        if groups:
            last = groups[-1]
            last_y = float(np.mean([f.y for f in last]))
            same_band = band_of(frag.y) is not None and band_of(frag.y) == band_of(last_y)
            if abs(frag.y - last_y) <= 0.6 * height or same_band:
                last.append(frag)
                continue
        groups.append([frag])
    groups = [part for group in groups for part in _split_merged_band(group, height)]
    lines = []
    for group in groups:
        cells, conf = [""] * len(COLUMNS), [100.0] * len(COLUMNS)
        struck: dict[int, list[str]] = {}
        for frag in sorted(group, key=lambda f: f.y):
            cells[frag.col] = f"{cells[frag.col]} {frag.text}".strip()
            conf[frag.col] = min(conf[frag.col], frag.conf)
            if frag.struck:
                struck.setdefault(frag.col, []).extend(frag.struck)
        names = [f for f in group if f.col == COL_NAME]
        name_x = (min(f.x0 for f in names), max(f.x1 for f in names)) if names else (0.0, 0.0)
        boxes = [(f.col, f.x0, f.y - f.height / 2, f.x1, f.y + f.height / 2) for f in group]
        lines.append(TableLine(y=float(np.mean([f.y for f in group])), cells=cells, conf=conf,
                               struck=struck, name_x=name_x, boxes=boxes))
    return lines


def _split_merged_band(group: list[Fragment], height: float) -> list[list[Fragment]]:
    """Полоса между линиями — несколько строк таблицы: в ней несколько единиц или несколько
    незачёркнутых количеств на разной высоте (разбивка «в том числе», не найден разделитель).
    Тогда делим по строкам.

    Зачёркнутое старое количество над новым — не повод: это правка в клетке."""
    def spread(ys: list[float]) -> bool:
        return len(ys) >= 2 and max(ys) - min(ys) > 1.2 * height

    units = [f.y for f in group if f.col == COL_UNIT]
    quantities = [f.y for f in group if f.col == COL_QTY and f.text]
    if not (spread(units) or spread(quantities)):
        return [group]
    parts: list[list[Fragment]] = []
    for frag in sorted(group, key=lambda f: f.y):
        if parts and abs(frag.y - float(np.mean([f.y for f in parts[-1]]))) <= 0.6 * height:
            parts[-1].append(frag)
        else:
            parts.append([frag])
    return parts


def mark_colored(lines: list[TableLine], located: Located) -> None:
    """Пометить строки с цветными чернилами (не меньше ~2 мм²) в графах единицы, количества
    и массы: там правка ручкой меняет значение. Штампы в «Примечании» значение не меняют."""
    grid = located.grid
    for line in lines:
        y0, y1 = int(line.y - 3 * MM), int(line.y + 3 * MM)
        area = located.colored[max(y0, 0):y1, grid.xs[COL_UNIT]:grid.xs[COL_MASS + 1]]
        line.colored = int((area > 0).sum()) > 2 * MM * MM


FOREIGN_INK_MM2 = 2.0  # непонятных чернил у чисел больше этого — штамп, рука или облако правки
_NUMBER_COLUMNS = (COL_UNIT, COL_QTY, COL_MASS)


def mark_foreign_ink(lines: list[TableLine], clean: np.ndarray, grid: Grid) -> None:
    """Пометить строки, где в графах единицы, количества и массы есть чернила, которые не
    объясняются распознанным текстом: штамп поверх числа, рукописная правка, облако
    изменения. Совпадение двух прочтений в таком месте ничего не доказывает."""
    left, right = grid.xs[COL_UNIT] + 8, grid.xs[COL_MASS + 1] - 8
    for line in lines:
        numbers = [b for b in line.boxes if b[0] in _NUMBER_COLUMNS]
        if not numbers:
            continue
        ys = [v for b in line.boxes for v in (b[2], b[4])]
        y0, y1 = int(min(ys)) - 6, int(max(ys)) + 6
        ink = clean[max(y0, grid.top):min(y1, grid.bottom), left:right] < 140
        if not ink.size:
            continue
        ink = ink.copy()
        for _col, bx0, by0, bx1, by1 in numbers:  # распознанный текст «объясняет» свои чернила
            ink[max(int(by0) - 6 - y0, 0):max(int(by1) + 6 - y0, 0),
                max(int(bx0) - 6 - left, 0):max(int(bx1) + 6 - left, 0)] = False
        line.foreign_ink = int(ink.sum()) > FOREIGN_INK_MM2 * MM * MM


def mark_headers(lines: list[TableLine], located: Located) -> None:
    """Отметить строки, оформленные как заголовок раздела: с номером («2.4. Крепёжные…»),
    подчёркнутые (линия под текстом короче графы) или стоящие по центру графы."""
    grid = located.grid
    left, right = grid.xs[COL_NAME], grid.xs[COL_NAME + 1]
    width = right - left
    for line in lines:
        name = line.cells[COL_NAME]
        if not name:
            continue
        if _SECTION_NUM_RE.match(name):
            line.header_like = True
            continue
        band = located.lines[int(line.y + 1 * MM):int(line.y + 4 * MM), left + 10:right - 10]
        coverage = int((band > 0).sum(axis=1).max()) if band.size else 0
        underlined = 8 * MM <= coverage < 0.9 * (width - 20)
        x0, x1 = line.name_x
        gap_left, gap_right = x0 - left, right - x1
        centered = gap_left > 0.12 * width and abs(gap_left - gap_right) < 0.25 * (gap_left + gap_right)
        line.header_like = underlined or centered


def read_table(located: Located, workers: int = 4, templates: UnitTemplates | None = None) -> list[TableLine]:
    """Распознать тело таблицы: графы параллельно (Tesseract — отдельные процессы).

    templates — образцы единиц документа: пополняются уверенными прочтениями и помогают
    прочитать нечитаемые единицы на этой и следующих страницах."""
    from concurrent.futures import ThreadPoolExecutor

    clean = erase_lines(located.img, located.lines)

    def column(col: int) -> list[Fragment]:
        if col == COL_UNIT:
            return ocr_unit_column(clean, located.grid)
        return ocr_column(clean, located.grid, col, original=located.img)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(column, range(len(COLUMNS))))
    fragments = [f for part in parts for f in part]
    resolve_units_by_templates(fragments, templates if templates is not None else {})
    fragments = [f for f in fragments if f.text or f.struck]  # пустые единицы (только картинка) — нет
    lines = assemble_lines(fragments, located.grid)
    mark_colored(lines, located)
    mark_foreign_ink(lines, clean, located.grid)
    mark_headers(lines, located)
    return lines


# ------------------------------ позиции -----------------------------------

# Единицы измерения в спецификациях. Распознанное значение сверяется со словарём:
# отличие в одну букву исправляется («шл.» → «шт.»), иначе значение помечается.
UNITS = ("шт.", "шт", "м", "м2", "м3", "м²", "м³", "кг", "т", "км", "л", "компл.", "компл", "к-т", "место", "п.м",
         "пара", "уп.", "рул.", "лист")
_UNIT_LOOKALIKES = str.maketrans({"з": "3", "З": "3"})  # «мЗ» → «м3»


def ocr_unit_column(clean: np.ndarray, grid: Grid) -> list[Fragment]:
    """Графа единиц: каждую строку отдельно и со сверкой со словарём.

    Tesseract плохо читает одиночные короткие слова («шт» → «ил», «НЯ»), поэтому строка
    распознаётся как одна строка текста, при неудаче — в увеличенном виде. Значение не
    из словаря остаётся как прочитано, с низкой уверенностью (→ «Проверить»)."""
    import pytesseract
    from PIL import Image

    x0, x1 = grid.xs[COL_UNIT] + CELL_PAD, grid.xs[COL_UNIT + 1] - CELL_PAD
    column = clean[grid.top:grid.bottom, x0:x1]
    ink = (column < 140).sum(axis=1) > 0
    fragments = []
    for a, b in _runs(ink, gap=4):
        if b - a < 1 * MM:  # точки, обрывки линий
            continue
        img = Image.fromarray(np.pad(column[a:b + 1], 20, constant_values=255))
        big = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        readings = [pytesseract.image_to_string(im, lang="rus", config="--psm 7").strip() for im in (img, big)]
        unit = _agreed_unit(readings)
        y = grid.top + (a + b) / 2
        ink = _trim(column[a:b + 1] < 140)
        cols = np.flatnonzero((column[a:b + 1] < 140).any(axis=0))
        ux0, ux1 = (x0 + int(cols[0]), x0 + int(cols[-1]) + 1) if len(cols) else (x0, x1)
        if unit:
            fragments.append(Fragment(COL_UNIT, y, b - a, unit, AGREED_CONFIDENCE, x0=ux0, x1=ux1, image=ink))
        elif any(readings) or ink is not None:
            fragments.append(Fragment(COL_UNIT, y, b - a, next((r for r in readings if r), ""), 20.0,
                                      x0=ux0, x1=ux1, image=ink))
    return fragments


def _trim(ink: np.ndarray) -> np.ndarray | None:
    """Обрезать картинку по чернилам (None — пусто)."""
    ys, xs = np.nonzero(ink)
    if not len(xs):
        return None
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


UnitTemplates = dict  # единица → список картинок, уверенно прочитанных в этом документе
MAX_TEMPLATES = 12  # образцов на единицу
TEMPLATE_MATCH = 0.72  # сходство с образцом, достаточное для единицы
TEMPLATE_MARGIN = 0.08  # и насколько лучше ближайшей другой единицы


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Сходство двух надписей (0..1): одинаковая высота, близкая ширина, совпадение чернил."""
    import cv2

    ha, wa = a.shape
    hb, wb = b.shape
    if not (0.75 < (wa / ha) / (wb / hb) < 1.33):
        return 0.0
    size = (48, max(8, int(48 * wa / ha)))
    ra = cv2.resize(a.astype(np.uint8) * 255, size[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    rb = cv2.resize(b.astype(np.uint8) * 255, size[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom else 0.0


def resolve_units_by_templates(fragments: list[Fragment], templates: UnitTemplates) -> None:
    """Нечитаемая единица сравнивается с уверенно прочитанными в этом же документе.

    Шрифт и качество скана внутри документа одни: если картинка «ил.» почти совпадает
    с образцами «шт.» и заметно хуже с остальными — это «шт.». Иначе остаётся неуверенной."""
    units = [f for f in fragments if f.col == COL_UNIT and f.image is not None]
    for frag in units:
        if frag.conf >= AGREED_CONFIDENCE:
            bucket = templates.setdefault(frag.text.rstrip("."), [])
            if len(bucket) < MAX_TEMPLATES:
                bucket.append(frag.image)
    for frag in units:
        if frag.conf >= AGREED_CONFIDENCE or not templates:
            continue
        scores = sorted(
            ((max(_similarity(frag.image, t) for t in images), unit) for unit, images in templates.items() if images),
            reverse=True,
        )
        best, unit = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        if best >= TEMPLATE_MATCH and best - second >= TEMPLATE_MARGIN:
            frag.text = next((u for u in UNITS if u.rstrip(".") == unit and u.endswith(".")), unit)
            frag.conf = AGREED_CONFIDENCE


def _agreed_unit(readings: list[str]) -> str:
    """Единица, в которой уверены: прочитана точно многобуквенной («шт», «км») либо оба
    прочтения сошлись. Однобуквенные «т», «л» — частые ошибки чтения «шт», им одного
    прочтения мало."""
    units = [normalize_unit(r) for r in readings]
    for raw, (unit, sure) in zip(readings, units):
        if sure and len(unit.rstrip(".")) >= 2 and raw.strip().lower().replace(",", ".") == unit:
            return unit
    sure_units = {u.rstrip(".") for u, ok in units if ok and u}
    if len(sure_units) == 1 and all(ok and u for u, ok in units):
        return next(u for u, ok in units)
    return ""

_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?\*?$")
_SECTION_NUM_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z]")
_CONTINUATION_START = ("ТУ", "ГОСТ", "ОСТ", "СТО", "по ", "с ", "в ", "из ", "и ", "для ")
_VARIANT_RE = re.compile(r"^(?:[A-ZА-ЯЁ]{1,3}[-\s]?\d|\d|[xх×]\d)")
MAX_VARIANT_WORDS = 3  # «Д-225х13,4 Ду200» — вариант; «325х8,0 ГОСТ… Труба стальная» — полное наименование


def _is_variant(name: str) -> bool:
    """Строка — только обозначение варианта без своего наименования."""
    return bool(_VARIANT_RE.match(name)) and len(name.split()) <= MAX_VARIANT_WORDS


@dataclass
class Position:
    """Позиция спецификации (одна или несколько строк таблицы)."""

    page: int  # страница файла, с 1
    section: str = ""
    cells: list[str] = field(default_factory=lambda: [""] * len(COLUMNS))
    issues: list[str] = field(default_factory=list)  # почему стоит проверить вручную
    struck: dict[int, list[str]] = field(default_factory=dict)  # зачёркнутое по графам
    changes: list[str] = field(default_factory=list)  # учтённые правки по документу (зачёркнуто → новое)
    subitem: bool = False  # строка «в том числе» — входит в позицию выше, в сумму не добавлять
    excluded: bool = False  # позиция зачёркнута целиком (исключена изменением)

    def add(self, line: TableLine) -> None:
        for col, text in enumerate(line.cells):
            if text:
                self.cells[col] = f"{self.cells[col]} {text}".strip()
        for col, words in line.struck.items():
            self.struck.setdefault(col, []).extend(words)

    def flag(self, reason: str) -> None:
        if reason not in self.issues:
            self.issues.append(reason)

    @property
    def has_quantity(self) -> bool:
        return bool(self.cells[COL_QTY] or self.cells[COL_UNIT])


def normalize_unit(raw: str) -> tuple[str, bool]:
    """Единица по словарю: (значение, уверенно ли). «М» → «м», «шл.» → «шт.»."""
    text = raw.strip().lower().replace(" ", "").replace(",", ".").translate(_UNIT_LOOKALIKES)
    if not text:
        return "", True
    if text in UNITS:
        return text, True
    if "/" in text:  # составные: «м/м2», «шт./м2» — каждая часть из словаря
        parts = [normalize_unit(part) for part in text.split("/")]
        if len(parts) == 2 and all(ok and unit for unit, ok in parts):
            return "/".join(unit for unit, _ in parts), True
    # Исправлять в одну букву можно только многобуквенные: иначе «ил» стало бы «л» (литры).
    close = [u for u in UNITS if len(u.rstrip(".")) >= 2 and _distance(text, u) <= 1]
    if len(set(u.rstrip(".") for u in close)) == 1:
        return close[0], True
    return raw.strip(), False


def _distance(a: str, b: str) -> int:
    """Расстояние Левенштейна (короткие строки)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_noise(line: TableLine) -> bool:
    """Строка-мусор: подписи, штампы, зачёркивания. Строки с количеством или номером позиции
    мусором не считаются никогда; текстовая — если в ней нет ни одной уверенной графы
    или ни одного слова из трёх букв."""
    if any(line.cells[c] for c in (COL_POS, COL_UNIT, COL_QTY, COL_MASS)):
        return False
    filled = [c for c, text in enumerate(line.cells) if text]
    if not filled:  # всё зачёркнуто или пусто
        return True
    if max(line.conf[c] for c in filled) < 50:
        return True
    words = " ".join(line.cells[c] for c in filled).split()
    return not any(len(re.sub(r"[^А-Яа-яЁёA-Za-z]", "", w)) >= 3 for w in words)


def _continues(text: str) -> bool:
    """Строка продолжает предыдущее наименование, а не начинает новое.

    Нумерованный заголовок («2.4. Крепёжные изделия») не продолжение, хотя начинается с цифры."""
    first = text[:1]
    if not first or _SECTION_NUM_RE.match(text):
        return False
    return not first.isalpha() or first.islower() or text.startswith(_CONTINUATION_START)


def _group_name(name: str) -> str:
    """Общая часть наименования без хвоста-обозначения: «Хомут для труб ПЭ Д-110» → «Хомут для труб ПЭ»."""
    words = name.split()
    while len(words) > 2 and re.search(r"\d", words[-1]):
        words.pop()
    return " ".join(words)


_BREAKDOWN_RE = re.compile(r"в\s+т(?:ом|\.)\s*ч(?:исле|\.)\s*:?", re.IGNORECASE)


def _is_heading(line: TableLine) -> bool:
    """Строка оформлена как заголовок раздела (см. mark_headers) или пронумерована как он."""
    return line.header_like or bool(_SECTION_NUM_RE.match(line.cells[COL_NAME]))


def group_positions(lines: list[TableLine], page: int, section: str = "") -> tuple[list[Position], str]:
    """Собрать строки таблицы в позиции. Возвращает позиции и раздел на конец страницы.

    Правила (форма ГОСТ 21.110):
    - строка с количеством или единицей — позиция; если у открытой позиции (начатой
      номером) количества ещё нет, строка дополняет её;
    - строка с номером в графе «Позиция» начинает позицию;
    - строка только с текстом продолжает открытую позицию, если начинается со строчной
      буквы, не с буквы или с «ТУ/ГОСТ…», либо если предыдущее наименование оборвано
      запятой. Иначе решение откладывается до следующей позиции: если та продолжает
      текст (со строчной, вариант «Д-225»), отложенные строки — начало её наименования;
      если нет — строки, оформленные как заголовок (подчёркнуты, по центру, с номером),
      становятся разделом, остальные дописываются к предыдущей позиции (параметры
      «Рабочее давление — 1,6 МПа», «Температура среды…»);
    - строка-вариант («Д-225», «3х2,5») без своего наименования получает общую часть
      наименования предыдущей позиции;
    - после «в том числе:» строки с количеством без единицы и номера — разбивка позиции
      выше (subitem): в сумму их добавлять нельзя."""
    positions: list[Position] = []
    current: Position | None = None
    pending: list[TableLine] = []
    group = ""
    breakdown: Position | None = None  # позиция с «в том числе:», чью разбивку читаем

    def settle_pending() -> None:
        """Отложенные строки не стали началом наименования: параметры — к предыдущей позиции,
        заголовки (и всё после первого заголовка) — в раздел."""
        nonlocal section, pending
        first_header = next((i for i, ln in enumerate(pending) if _is_heading(ln)), len(pending))
        if current is None:
            first_header = 0
        for ln in pending[:first_header]:
            current.add(ln)
        heading = pending[first_header:]
        if heading:
            section = " / ".join(" ".join(c for c in ln.cells if c) for ln in heading)
        pending = []

    for line in lines:
        name = line.cells[COL_NAME]
        has_qty = bool(line.cells[COL_QTY] or line.cells[COL_UNIT] or line.cells[COL_MASS])
        if line.struck.get(COL_NAME) and not name:
            # наименование зачёркнуто целиком — позиция исключена изменением (см. _apply_strikes)
            settle_pending()
            item = Position(page=page, section=section)
            item.add(line)
            positions.append(item)
            current, breakdown = item, None
            continue
        if _is_noise(line) and not line.struck:
            # подписи и штампы в «Примечании» и у завода значения не трогают; рядом с
            # наименованием или маркой — возможна вписанная от руки правка
            if current is not None and (line.cells[COL_NAME] or line.cells[COL_TYPE]):
                current.flag("рядом нераспознанный текст (возможна правка от руки)")
            continue
        sub = (breakdown is not None and has_qty and not line.cells[COL_UNIT] and not line.cells[COL_POS]
               and not pending)
        if sub:
            item = Position(page=page, section=section, subitem=True)
            item.cells[COL_NAME] = _BREAKDOWN_RE.sub("", group).strip(" :")
            item.add(line)
            positions.append(item)
            current = item
            continue
        if has_qty and current is not None and not current.has_quantity and not line.cells[COL_POS]:
            current.add(line)
        elif has_qty or line.cells[COL_POS]:
            item = Position(page=page)
            if pending and (not name or _continues(name) or _is_variant(name)):
                # начало наименования на строках выше; заголовки перед ним — раздел
                first_text = next((i for i, ln in enumerate(pending) if not _is_heading(ln)), len(pending))
                if first_text:
                    section = " / ".join(" ".join(c for c in ln.cells if c) for ln in pending[:first_text])
                for ln in pending[first_text:]:
                    item.add(ln)
                pending = []
            else:
                settle_pending()
                if name and _is_variant(name) and group:
                    item.cells[COL_NAME] = group
            item.section = section
            item.add(line)
            positions.append(item)
            current = item
            if name and not _is_variant(name):
                group = _group_name(item.cells[COL_NAME])
            breakdown = item if _BREAKDOWN_RE.search(item.cells[COL_NAME]) else (
                breakdown if _is_variant(name or "") else None)
        elif current is not None and not pending and (
            _continues(name or " ".join(line.cells)) or current.cells[COL_NAME].endswith((",", "-"))
        ):
            current.add(line)
            if current is not None and not current.subitem and _BREAKDOWN_RE.search(name):
                breakdown = current
                group = _group_name(current.cells[COL_NAME])
        else:
            pending.append(line)
            breakdown = None
        if current is not None and not pending:
            if line.colored:
                current.flag("цветные пометки у количества или массы (ручка, штамп)")
            elif line.foreign_ink:
                current.flag("посторонние пометки у чисел (штамп, рука, облако правки)")
            unsure = [label for col, label in ((COL_UNIT, "ед."), (COL_QTY, "кол-во"), (COL_MASS, "масса"))
                      if line.cells[col] and line.conf[col] < LOW_CONFIDENCE]
            if unsure:
                current.flag("неуверенно прочитано: " + ", ".join(unsure))
            if line.cells[COL_NAME] and line.conf[COL_NAME] < POOR_NAME_CONFIDENCE:
                current.flag("наименование прочитано плохо")
    settle_pending()
    for item in positions:
        _check(item)
    return positions, section


# Размеры «114х5», «325х8,0»: в окружении кириллицы OCR путает цифры с похожими буквами.
_DIMENSION_RE = re.compile(r"(?<![\wА-Яа-яЁё])(\d+[.,]?\d*)([xх×Xх])([\dбБЗзОо]+(?:[.,][\dбБЗзОо]+)?)(?=[^\wА-Яа-яЁё]|$)")
_DIGIT_LOOKALIKES = str.maketrans({"б": "6", "Б": "6", "З": "3", "з": "3", "О": "0", "о": "0"})


def fix_dimensions(text: str) -> str:
    """Исправить похожие на цифры буквы только внутри размеров: «114хб» → «114х6».

    Знак «х» в размерах приводится к кириллическому: OCR ставит то латинскую x, то
    кириллическую, а в реестре одинаковые позиции должны находиться фильтром одинаково."""
    return _DIMENSION_RE.sub(lambda m: m.group(1) + "х" + m.group(3).translate(_DIGIT_LOOKALIKES), text)


def _check(item: Position) -> None:
    """Проверки значений позиции: правки по документу, числа в количестве и массе, единица."""
    for col in (COL_NAME, COL_TYPE):
        item.cells[col] = fix_dimensions(item.cells[col])
    _apply_strikes(item)
    if item.excluded:
        return
    unit, sure = normalize_unit(item.cells[COL_UNIT])
    item.cells[COL_UNIT] = unit
    if not sure:
        item.flag("единица не из словаря")
    for col, label in ((COL_QTY, "количество"), (COL_MASS, "масса")):
        value = item.cells[col].strip()
        if value and not _NUMBER_RE.match(value):  # «12 15» — зачёркнуто и вписано новое
            item.flag(f"{label}: не одно число (возможна правка)")
    if not item.cells[COL_QTY]:
        item.flag("нет количества")
    if not item.cells[COL_NAME]:
        item.flag("нет наименования")
    elif not item.subitem and not re.search(r"[А-Яа-яЁёA-Za-z]{3,}", item.cells[COL_NAME]):
        item.flag("наименование прочитано плохо")  # обрывки букв — рамка правки или штамп в клетке


def _apply_strikes(item: Position) -> None:
    """Зачёркнутое по документу: старое значение не берётся, правка записывается.

    Наименование зачёркнуто целиком — позиция исключена изменением. Зачёркнуто
    количество или масса и рядом новое число — берётся новое («было 210,4»). Зачёркнуто,
    а нового нет — позиция к проверке."""
    struck = item.struck
    if not struck:
        return
    if struck.get(COL_NAME) and not item.cells[COL_NAME]:
        item.excluded = True
        item.cells[COL_NAME] = " ".join(struck[COL_NAME])
        for col in (COL_TYPE, COL_QTY, COL_MASS):
            if struck.get(col) and not item.cells[col]:
                item.cells[col] = " ".join(struck[col])
        item.changes.append("позиция зачёркнута (исключена изменением)")
        return
    for col, label, lost in ((COL_QTY, "кол-во", "количество зачёркнуто"), (COL_MASS, "масса", "масса зачёркнута")):
        old = struck.get(col)
        if not old:
            continue
        if item.cells[col]:  # зачёркнутые цифры читаются хуже: старое значение — справочно
            item.changes.append(f"{label}: было (по скану) {' '.join(old)}")
        else:
            item.flag(f"{lost}, нового значения не найдено")
    if struck.get(COL_NAME):
        item.changes.append("в наименовании зачёркнуто: " + " ".join(struck[COL_NAME]))
    if struck.get(COL_TYPE):
        item.changes.append("в марке зачёркнуто: " + " ".join(struck[COL_TYPE]))


# ------------------------------ документы и реестр ------------------------

MAX_SPEC_FILES = 20  # как лимит файлов в чате
MAX_SPEC_PAGES = 150  # страниц за один реестр: ~2–3 с на страницу → до ~6 минут
MAX_LEADING_MISSES = 5  # столько страниц подряд в начале без формы — файл не спецификация

# Шифр документа в основной надписи: «12345678.001-Р-100.000.000-НВК-02-С-001». Начинается
# с номера (6+ цифр); буквы в шифрах кириллические, OCR же ставит латинские двойники.
_CODE_RE = re.compile(r"\d{6,9}[.,_\-][0-9A-ZА-ЯЁ.,_\-]{8,}")
_CODE_LOOKALIKES = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")


def clean_code(raw: str) -> str:
    """Шифр из OCR: без хвостовых букв из соседней клетки, латинские двойники → кириллица."""
    code = re.sub(r"[^\d]+$", "", raw.strip(".,-_ "))
    return code.translate(_CODE_LOOKALIKES)


def unavailable_reason() -> str | None:
    """Почему реестр спецификаций сейчас не построить (нет Tesseract или OpenCV), иначе None."""
    import shutil

    if not shutil.which("tesseract"):
        return "Распознавание сканов (Tesseract) на сервере не установлено — реестр спецификаций недоступен."
    try:
        import cv2  # noqa: F401
        import pytesseract  # noqa: F401
    except ImportError:
        return "На сервере не установлены библиотеки распознавания (opencv-python-headless, pytesseract)."
    return None


def check_pdf(content: bytes, filename: str) -> None:
    """Файл — PDF разумного размера (тот же лимит, что у остальных операций с PDF)."""
    from app.services.document_processing import MAX_FILE_SIZE_PDF, DocumentError

    if not filename.lower().endswith(".pdf"):
        raise DocumentError(f"«{filename}»: для реестра спецификаций нужен PDF")
    if len(content) > MAX_FILE_SIZE_PDF:
        raise DocumentError(
            f"«{filename}» слишком большой ({len(content) // (1024 * 1024)} МБ), "
            f"лимит — {MAX_FILE_SIZE_PDF // (1024 * 1024)} МБ."
        )


@dataclass
class SpecDocument:
    filename: str
    code: str = ""  # шифр из основной надписи (OCR); если не прочитан — имя файла
    pages: int = 0
    table_pages: int = 0
    skipped_pages: list[int] = field(default_factory=list)  # нет формы спецификации
    failed_pages: list[int] = field(default_factory=list)  # сбой разбора страницы (остальные разобраны)
    unchecked_pages: int = 0  # не просматривались (файл не похож на спецификацию)
    positions: list[Position] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None

    @property
    def flagged(self) -> int:
        return sum(1 for p in self.positions if p.issues)

    @property
    def changed(self) -> int:
        return sum(1 for p in self.positions if p.changes and not p.excluded)

    @property
    def excluded(self) -> int:
        return sum(1 for p in self.positions if p.excluded)


@dataclass
class SpecRegistry:
    documents: list[SpecDocument]
    seconds: float

    @property
    def positions(self) -> int:
        return sum(len(d.positions) for d in self.documents)

    @property
    def flagged(self) -> int:
        return sum(d.flagged for d in self.documents)

    @property
    def changed(self) -> int:
        return sum(d.changed for d in self.documents)

    @property
    def excluded(self) -> int:
        return sum(d.excluded for d in self.documents)


def read_code(located: Located) -> str:
    """Шифр документа из основной надписи под таблицей: самая длинная «кодовая» строка."""
    import pytesseract
    from PIL import Image

    grid, img = located.grid, located.img
    x0 = max(grid.xs[-1] - int(190 * MM), 0)
    area = img[grid.bottom:min(grid.bottom + int(60 * MM), img.shape[0]), x0:grid.xs[-1] + int(5 * MM)]
    if area.size == 0:
        return ""
    try:
        text = pytesseract.image_to_string(Image.fromarray(area), lang="rus+eng", config="--psm 6")
    except Exception:  # noqa: BLE001 — шифр необязателен: возьмём имя файла
        return ""
    candidates = [clean_code(m.group(0)) for m in _CODE_RE.finditer(text.upper().replace(" ", ""))]
    candidates = [c for c in candidates if c.count("-") >= 2]
    return max(candidates, key=len) if candidates else ""


def process_pdf(content: bytes, filename: str, page_budget: int) -> SpecDocument:
    """Разобрать один PDF: страницы с формой спецификации → позиции."""
    import time
    from collections import Counter

    import pymupdf

    started = time.time()
    doc_info = SpecDocument(filename=filename)
    try:
        pdf = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — битый файл не должен ронять весь реестр
        doc_info.error = f"не удалось открыть PDF: {exc}"
        return doc_info
    doc_info.pages = len(pdf)
    codes: Counter[str] = Counter()
    section = ""
    templates: UnitTemplates = {}
    for index in range(min(len(pdf), page_budget)):
        if doc_info.table_pages == 0 and len(doc_info.skipped_pages) >= MAX_LEADING_MISSES:
            doc_info.unchecked_pages = min(len(pdf), page_budget) - index
            break
        if too_large_for_form(pdf[index]):
            doc_info.skipped_pages.append(index + 1)
            continue
        gray, colored = render_page(pdf[index])
        try:
            located = locate_form(gray, colored)
        except SpecFormNotFound:
            doc_info.skipped_pages.append(index + 1)
            continue
        doc_info.table_pages += 1
        try:
            if len(codes) < 3:
                code = read_code(located)
                if code:
                    codes[code] += 1
            items, section = group_positions(read_table(located, templates=templates), index + 1, section)
        except Exception:  # noqa: BLE001 — сбой одной страницы не должен обрывать весь реестр
            logger.exception("spec registry: page %s of %s failed", index + 1, filename)
            doc_info.failed_pages.append(index + 1)
            continue
        doc_info.positions.extend(items)
    flag_unit_outliers(doc_info.positions)
    if len(pdf) > page_budget:
        doc_info.unchecked_pages += len(pdf) - page_budget
    doc_info.code = codes.most_common(1)[0][0] if codes else filename.rsplit(".", 1)[0]
    doc_info.seconds = round(time.time() - started, 1)
    return doc_info


def flag_unit_outliers(positions: list[Position]) -> None:
    """Редкая для документа единица, отличающаяся от основной одной буквой («км» среди «м»):
    так выглядит ошибка чтения (дуга облака правки или штрих прочитаны как буква)."""
    from collections import Counter

    units = Counter(p.cells[COL_UNIT].rstrip(".") for p in positions if p.cells[COL_UNIT] and not p.excluded)
    if not units:
        return
    main, main_count = units.most_common(1)[0]
    total = sum(units.values())
    if main_count < 0.5 * total:
        return
    for item in positions:
        unit = item.cells[COL_UNIT].rstrip(".")
        if unit and unit != main and units[unit] <= max(3, 0.15 * total) and _distance(unit, main) == 1:
            item.flag(f"единица «{item.cells[COL_UNIT]}» редкая для документа (обычно «{main}»)")


def build_spec_registry(files: list[tuple[str, bytes]]) -> SpecRegistry:
    """Реестр позиций по нескольким PDF. Страниц всего не больше MAX_SPEC_PAGES."""
    import time

    started = time.time()
    documents, budget = [], MAX_SPEC_PAGES
    for name, content in files:
        if budget <= 0:
            documents.append(SpecDocument(filename=name, error="не обработан: исчерпан лимит страниц на один реестр"))
            continue
        document = process_pdf(content, name, budget)
        budget -= document.table_pages + len(document.skipped_pages)
        documents.append(document)
    return SpecRegistry(documents=documents, seconds=round(time.time() - started, 1))


def _as_number(text: str):
    """«12,0» → 12.0 для Excel (суммы и фильтры работают); иначе текст как есть."""
    if re.fullmatch(r"\d+(?:[.,]\d+)?", text or ""):
        return float(text.replace(",", "."))
    return text


EXCEL_HEADERS = [
    "Шифр (по штампу)", "Файл", "Стр.", "Раздел", "Позиция", "Наименование и техническая характеристика",
    "Тип, марка, обозначение документа", "Код", "Завод-изготовитель", "Ед. изм.", "Кол-во",
    "Масса ед., кг", "В т. ч.", "Примечание", "Правки по документу", "Проверить",
]
EXCEL_WIDTHS = [26, 22, 6, 24, 9, 60, 28, 12, 20, 8, 9, 10, 7, 22, 28, 34]

HOW_TO_READ = [
    "Реестр собран по сканам спецификаций (форма ГОСТ 21.110) без участия языковой модели.",
    "Каждая графа распознана отдельно: количество и масса — только цифры, единицы сверены со словарём.",
    "Значения не додумываются: что не прочиталось уверенно, остаётся как прочитано и попадает в колонку «Проверить».",
    "Жёлтым выделены позиции с причинами для проверки: правки от руки, цветные пометки, неуверенно прочитанные числа, не число в количестве.",
    "Голубым — учтённые правки по документу: зачёркнутое значение заменено новым, старое указано в колонке «Правки по документу».",
    "Серым зачёркнутым шрифтом — позиции, зачёркнутые целиком (исключены изменением); в суммы их не включайте.",
    "«В т. ч.» = да — разбивка позиции выше («в том числе»); в суммы её не включайте, иначе количество посчитается дважды.",
    "Рукописные исправления система не читает: такие позиции выделены жёлтым, сверяйте их со сканом.",
    "«То же» в наименовании оставлено как в документе.",
    "Шифр прочитан из основной надписи распознаванием и может содержать ошибки; рядом — имя файла.",
    "Лист «Документы»: сколько страниц разобрано, на каких не найдена форма спецификации.",
]


def build_spec_xlsx(registry: SpecRegistry) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    from app.services.document_processing import append_row

    wb = Workbook()
    ws = wb.active
    ws.title = "Позиции"
    append_row(ws, EXCEL_HEADERS)
    head_fill = PatternFill("solid", fgColor="E7EEF8")
    warn = PatternFill("solid", fgColor="FFF2B3")
    changed = PatternFill("solid", fgColor="DDEBF7")
    excluded = Font(color="808080", strike=True)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = head_fill
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    for doc in registry.documents:
        for item in doc.positions:
            c = item.cells
            append_row(ws, [
                doc.code, doc.filename, item.page, item.section, c[COL_POS], c[COL_NAME], c[COL_TYPE],
                c[COL_CODE], c[COL_MAKER], c[COL_UNIT], _as_number(c[COL_QTY]), _as_number(c[COL_MASS]),
                "да" if item.subitem else "", c[COL_NOTE], "; ".join(item.changes), "; ".join(item.issues),
            ])
            for cell in ws[ws.max_row]:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if item.issues:
                    cell.fill = warn
                elif item.changes:
                    cell.fill = changed
                if item.excluded:
                    cell.font = excluded
    for i, width in enumerate(EXCEL_WIDTHS, 1):
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    docs = wb.create_sheet("Документы")
    append_row(docs, ["Файл", "Шифр (по штампу)", "Страниц", "С таблицей", "Позиций", "Проверить",
                      "Без формы спецификации (стр.)", "Не просмотрено стр.", "Время, с", "Ошибка"])
    for cell in docs[1]:
        cell.font = Font(bold=True)
        cell.fill = head_fill
    for doc in registry.documents:
        error = doc.error or ""
        if doc.failed_pages:
            error = f"не разобраны стр. {', '.join(map(str, doc.failed_pages))} (сбой распознавания)"
        append_row(docs, [doc.filename, doc.code, doc.pages, doc.table_pages, len(doc.positions), doc.flagged,
                          ", ".join(map(str, doc.skipped_pages)), doc.unchecked_pages, doc.seconds, error])
    for i, width in enumerate([40, 34, 9, 11, 9, 10, 26, 12, 9, 40], 1):
        docs.column_dimensions[get_column_letter(i)].width = width

    note = wb.create_sheet("Как читать")
    for line in HOW_TO_READ:
        append_row(note, [line])
    note.column_dimensions["A"].width = 120

    import io

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
