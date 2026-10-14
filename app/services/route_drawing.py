"""План трассы трубопровода: описание участков → эскизный чертёж (DXF, PDF, PNG) через CADCraft.

Тот же приём, что у конструктора деталей (services/cad.py): модель (или пользователь) задаёт
небольшую схему — участки трассы с длиной и поворотом, диаметр трубы, а команды для чертёжной
программы строит наш код. В программу попадают только проверенные числа, а результат сверяется
с расчётом: сумма длин участков, число размеров, габариты в пределах рамки.

CADCraft — сторонний открытый проект (MIT/Apache-2.0, github.com/storytold/cadcraft), запускается
как отдельная программа `cadcraft-cli`. Чертёж — ЭСКИЗ: условный план без привязки к местности,
отметок и пересечений; рамка и штамп похожи на ГОСТ, но соответствие стандартам не заявляется.

Система координат чертежа — миллиметры на листе A3 (420 × 297); реальные длины пересчитываются
по стандартному масштабу, а подписи размеров берутся из расчёта, а не из измерения на листе.
"""

from __future__ import annotations

import base64
import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services import cad

MAX_SEGMENTS = 12
MIN_LENGTH_M = 0.5
MAX_LENGTH_M = 10_000.0
MIN_DIAMETER_MM = 10.0
MAX_DIAMETER_MM = 2_000.0
CLI_TIMEOUT = 30

# Стандартные масштабы уменьшения (ГОСТ 2.302) и несколько мелких для длинных трасс.
SCALES = [2, 2.5, 4, 5, 10, 15, 20, 25, 40, 50, 75, 100, 200, 400, 500, 800, 1000, 2000, 2500, 5000, 10000]

# Лист A3 (мм): рамка по ГОСТ 2.104 — поле слева 20 мм, остальные по 5 мм.
SHEET_W, SHEET_H = 420.0, 297.0
FRAME = (20.0, 5.0, 415.0, 292.0)
STAMP_W, STAMP_H = 185.0, 40.0  # упрощённая основная надпись в правом нижнем углу
# Поле чертежа: слева от ведомости участков, над основной надписью.
AREA = (28.0, STAMP_H + 14.0, 292.0, 284.0)
DIM_OFFSET = 14.0  # отступ размерной линии от оси трассы
ANGLE_RADIUS = 9.0  # радиус дуги размера угла
EDGE_MARGIN = 14.0  # запас у края поля под размеры и подписи
STRAIGHT_TOLERANCE_DEG = 0.5  # поворот меньше этого — продолжение по прямой

# Слои названы латиницей: в DXF/DWG версии R2000 кириллица в именах записывается кодами \U+0420…,
# и не каждая программа показывает их как буквы.
LAYER_FRAME = "FRAME"
LAYER_PIPE = "PIPELINE"
LAYER_DIMS = "DIMENSIONS"
LAYER_TEXT = "TEXT"


class CadcraftError(Exception):
    """Ошибка построения чертежа с текстом для пользователя."""


class CadcraftUnavailable(CadcraftError):
    """CADCraft не установлен (scripts/fetch_cadcraft.sh)."""


class RouteSpecError(CadcraftError):
    """Схема трассы не прошла проверку."""


class Segment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    length_m: float
    # Поворот перед участком относительно прежнего направления, градусы:
    # положительный — влево (против часовой стрелки), отрицательный — вправо. У первого участка не используется.
    turn_deg: float = 0.0


class RouteSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = "План трассы трубопровода"
    diameter_mm: float
    wall_mm: float | None = None
    material: str = "сталь"
    segments: list[Segment] = Field(default_factory=list)
    # Что из описания чертёж не умеет (отметки, глубина заложения…): чертёж строится без этого.
    skipped: list[str] = Field(default_factory=list)


@dataclass
class Geometry:
    """Вершины трассы в метрах (первая вершина — начало, ось X вправо), длины и повороты в вершинах."""

    points: list[tuple[float, float]]
    headings: list[float]  # направление каждого участка, градусы
    total_m: float
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


def route_geometry(spec: RouteSpec) -> Geometry:
    x = y = 0.0
    heading = 0.0
    points = [(x, y)]
    headings: list[float] = []
    total = 0.0
    for index, segment in enumerate(spec.segments):
        if index > 0:
            heading += segment.turn_deg
        headings.append(heading)
        x += segment.length_m * math.cos(math.radians(heading))
        y += segment.length_m * math.sin(math.radians(heading))
        points.append((x, y))
        total += segment.length_m
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return Geometry(points, headings, total, (min(xs), min(ys), max(xs), max(ys)))


def _segments_cross(a1, a2, b1, b2) -> bool:
    """Пересекаются ли отрезки (в том числе касаются концом или перекрываются)."""

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):  # r лежит в охватывающем прямоугольнике отрезка pq
        return (min(p[0], q[0]) - 1e-9 <= r[0] <= max(p[0], q[0]) + 1e-9
                and min(p[1], q[1]) - 1e-9 <= r[1] <= max(p[1], q[1]) + 1e-9)

    o1, o2, o3, o4 = orient(a1, a2, b1), orient(a1, a2, b2), orient(b1, b2, a1), orient(b1, b2, a2)
    if o1 * o2 < -1e-9 and o3 * o4 < -1e-9:
        return True
    return ((abs(o1) <= 1e-9 and on_segment(a1, a2, b1)) or (abs(o2) <= 1e-9 and on_segment(a1, a2, b2))
            or (abs(o3) <= 1e-9 and on_segment(b1, b2, a1)) or (abs(o4) <= 1e-9 and on_segment(b1, b2, a2)))


def validate_spec(spec: RouteSpec) -> list[str]:
    errors: list[str] = []
    if not (MIN_DIAMETER_MM <= spec.diameter_mm <= MAX_DIAMETER_MM):
        errors.append(f"диаметр трубы должен быть от {MIN_DIAMETER_MM:g} до {MAX_DIAMETER_MM:g} мм")
    if spec.wall_mm is not None and not (0 < spec.wall_mm < spec.diameter_mm / 2):
        errors.append("толщина стенки должна быть больше 0 и меньше радиуса трубы")
    if not spec.segments:
        errors.append("не указано ни одного участка трассы")
    elif len(spec.segments) > MAX_SEGMENTS:
        errors.append(f"участков слишком много ({len(spec.segments)}), допустимо не более {MAX_SEGMENTS}")
    if errors:
        return errors
    for number, segment in enumerate(spec.segments, start=1):
        if not (MIN_LENGTH_M <= segment.length_m <= MAX_LENGTH_M):
            errors.append(f"участок {number}: длина {segment.length_m:g} м вне допустимого диапазона "
                          f"{MIN_LENGTH_M:g}–{MAX_LENGTH_M:g} м")
        if number > 1 and not (abs(segment.turn_deg) < 180):
            errors.append(f"участок {number}: поворот {segment.turn_deg:g}° должен быть по модулю меньше 180° "
                          "(0° — продолжение по прямой)")
    if errors:
        return errors
    points = route_geometry(spec).points
    for i in range(len(points) - 1):
        for j in range(i + 2, len(points) - 1):
            if _segments_cross(points[i], points[i + 1], points[j], points[j + 1]):
                errors.append(f"трасса пересекает сама себя: участки {i + 1} и {j + 1}")
                return errors
    return errors


def choose_scale(geometry: Geometry) -> int | float:
    """Наименьший стандартный масштаб 1:N, при котором трасса с запасом под размеры помещается в поле."""
    width_m = geometry.bbox[2] - geometry.bbox[0]
    height_m = geometry.bbox[3] - geometry.bbox[1]
    free_w = AREA[2] - AREA[0] - 2 * EDGE_MARGIN
    free_h = AREA[3] - AREA[1] - 2 * EDGE_MARGIN
    for scale in SCALES:
        if width_m * 1000 / scale <= free_w and height_m * 1000 / scale <= free_h:
            return scale
    raise RouteSpecError("трасса слишком велика даже для самого мелкого масштаба 1:10000")


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _scale_label(scale: int | float) -> str:
    return f"1:{scale:g}"


def paper_points(spec: RouteSpec, geometry: Geometry, scale: int | float) -> list[tuple[float, float]]:
    """Вершины трассы в мм на листе: по центру поля чертежа."""
    xmin, ymin, xmax, ymax = geometry.bbox
    cx = (AREA[0] + AREA[2]) / 2 - (xmin + xmax) / 2 * 1000 / scale
    cy = (AREA[1] + AREA[3]) / 2 - (ymin + ymax) / 2 * 1000 / scale
    return [(cx + px * 1000 / scale, cy + py * 1000 / scale) for px, py in geometry.points]


def _grid(add, x: float, y_top: float, col_widths: list[float], row_h: float, rows: list[list[str]],
          *, text_h: float = 3.0, merge_first_row: bool = False) -> None:
    """Таблица из линий и текста: надёжнее встроенной команды table (мелкий текст в ячейках).

    `merge_first_row` — первая строка без вертикальных разделителей, кроме после первой колонки.
    """
    width = sum(col_widths)
    height = row_h * len(rows)
    for i in range(len(rows) + 1):
        add("line", points=[[x, y_top - i * row_h], [x + width, y_top - i * row_h]])
    edges = [0.0]
    for w in col_widths:
        edges.append(edges[-1] + w)
    for j, offset in enumerate(edges):
        if merge_first_row and 1 < j < len(edges) - 1:
            add("line", points=[[x + offset, y_top - row_h], [x + offset, y_top - height]])
        else:
            add("line", points=[[x + offset, y_top], [x + offset, y_top - height]])
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if cell:
                add("text", at=[round(x + edges[j] + 1.5, 3), round(y_top - (i + 0.5) * row_h, 3)],
                    text=cell, height=text_h, justify="ML")


def _bend_count(spec: RouteSpec) -> int:
    return sum(1 for segment in spec.segments[1:] if abs(segment.turn_deg) > STRAIGHT_TOLERANCE_DEG)


def build_commands(spec: RouteSpec) -> tuple[list[str], dict]:
    """Команды CADCraft (`id {json}`) и сведения для проверки."""
    geometry = route_geometry(spec)
    scale = choose_scale(geometry)
    pts = paper_points(spec, geometry, scale)
    cmds: list[str] = []

    def add(command: str, **params) -> None:
        cmds.append(f"{command} {json.dumps(params, ensure_ascii=False)}")

    # --- слои и стиль размеров
    add("layer.new", name=LAYER_FRAME, lineweight=0.7)
    add("layer.new", name=LAYER_PIPE, lineweight=0.8)
    add("layer.new", name=LAYER_DIMS, lineweight=0.25)
    add("layer.new", name=LAYER_TEXT, lineweight=0.25)
    add("style", name="Standard", widthFactor=0.7, current=True)  # шрифт по умолчанию очень широкий
    add("dimstyle", name="Standard", current=True, DIMTXT=3.5, DIMASZ=3.0, DIMEXE=2.0, DIMEXO=1.5,
        DIMGAP=1.0, DIMTAD=1, DIMDEC=0, DIMADEC=0)

    # --- рамка листа и основная надпись
    add("layer.current", name=LAYER_FRAME)
    add("rectang", p1=[0, 0], p2=[SHEET_W, SHEET_H])
    add("rectang", p1=[FRAME[0], FRAME[1]], p2=[FRAME[2], FRAME[3]], width=0.7)
    stamp_x, stamp_y = FRAME[2] - STAMP_W, FRAME[1]
    total_text = _fmt(geometry.total_m) + " м"
    pipe_text = f"Ø{spec.diameter_mm:g}" + (f"×{spec.wall_mm:g}" if spec.wall_mm else "") + f", {spec.material}"
    add("layer.current", name=LAYER_TEXT)
    _grid(add, stamp_x, stamp_y + STAMP_H, [44, 58, 32, 51], STAMP_H / 5,
          [["Наименование", spec.title, "", ""],
           ["Труба", pipe_text, "Масштаб", _scale_label(scale)],
           ["Длина трассы", total_text, "Лист", "1"],
           ["Участков", str(len(spec.segments)), "Листов", "1"],
           ["Разработал", "", "Дата", ""]], text_h=2.4, merge_first_row=True)

    # --- трасса: условное обозначение трубопровода (жирная линия, не в масштабе диаметра)
    add("layer.current", name=LAYER_PIPE)
    add("pline", vertices=[[round(x, 4), round(y, 4)] for x, y in pts], width=0.8)

    # --- узловые точки с номерами: у внутренних — по внешней биссектрисе, у концов — вдоль продолжения трассы
    add("layer.current", name=LAYER_TEXT)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    for number, (x, y) in enumerate(pts, start=1):
        add("circle", center=[round(x, 4), round(y, 4)], radius=1.2)
        if 1 < number < len(pts):
            a, b = pts[number - 2], pts[number]
            u1 = (a[0] - x, a[1] - y)
            u2 = (b[0] - x, b[1] - y)
            n1, n2 = math.hypot(*u1), math.hypot(*u2)
            bis = (u1[0] / n1 + u2[0] / n2, u1[1] / n1 + u2[1] / n2)
            bn = math.hypot(*bis)
            if bn < 1e-6:  # участки на одной прямой: смещаем в сторону
                bis, bn = (-u2[1] / n2, u2[0] / n2), 1.0
            ox, oy = -bis[0] / bn * 6, -bis[1] / bn * 6
        else:
            other = pts[1] if number == 1 else pts[-2]
            dx, dy = x - other[0], y - other[1]
            norm = math.hypot(dx, dy) or 1.0
            ox, oy = dx / norm * 5, dy / norm * 5
        add("text", at=[round(x + ox, 3), round(y + oy, 3)], text=str(number), height=3.5, justify="MC")

    # --- размеры участков (подписи — реальные длины из расчёта) и углы поворотов
    add("layer.current", name=LAYER_DIMS)
    for number, (a, b) in enumerate(zip(pts, pts[1:]), start=1):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if (mx - cx) * nx + (my - cy) * ny < 0:  # размер выносим наружу от центра трассы
            nx, ny = -nx, -ny
        add("dimaligned", p1=[round(a[0], 4), round(a[1], 4)], p2=[round(b[0], 4), round(b[1], 4)],
            at=[round(mx + nx * DIM_OFFSET, 4), round(my + ny * DIM_OFFSET, 4)],
            text=_fmt(spec.segments[number - 1].length_m))
    for i in range(1, len(pts) - 1):
        if abs(spec.segments[i].turn_deg) <= STRAIGHT_TOLERANCE_DEG:
            continue  # продолжение по прямой: угла нет
        v, p1, p2 = pts[i], pts[i - 1], pts[i + 1]
        u1 = ((p1[0] - v[0]), (p1[1] - v[1]))
        u2 = ((p2[0] - v[0]), (p2[1] - v[1]))
        n1, n2 = math.hypot(*u1), math.hypot(*u2)
        bis = (u1[0] / n1 + u2[0] / n2, u1[1] / n1 + u2[1] / n2)
        bn = math.hypot(*bis) or 1.0
        add("dimangular", vertex=[round(v[0], 4), round(v[1], 4)],
            p1=[round(v[0] + u1[0] / n1 * 10, 4), round(v[1] + u1[1] / n1 * 10, 4)],
            p2=[round(v[0] + u2[0] / n2 * 10, 4), round(v[1] + u2[1] / n2 * 10, 4)],
            at=[round(v[0] + bis[0] / bn * ANGLE_RADIUS, 4), round(v[1] + bis[1] / bn * ANGLE_RADIUS, 4)])

    # --- ведомость участков справа
    add("layer.current", name=LAYER_TEXT)
    rows = [["Уч.", "Длина, м", "Угол в т.", "Поворот"]]
    for n, segment in enumerate(spec.segments, start=1):
        if n == 1:
            rows.append([f"1–2", _fmt(segment.length_m), "—", "—"])
        else:
            straight = abs(segment.turn_deg) <= STRAIGHT_TOLERANCE_DEG
            rows.append([f"{n}–{n + 1}", _fmt(segment.length_m),
                         "180°" if straight else f"{_fmt(180 - abs(segment.turn_deg), 0)}°",
                         "—" if straight else ("влево" if segment.turn_deg > 0 else "вправо")])
    rows.append(["Σ", _fmt(geometry.total_m), "", ""])
    table_x = AREA[2] + 8
    add("text", at=[table_x, FRAME[3] - 4], text="Ведомость участков", height=3.0, justify="ML")
    _grid(add, table_x, FRAME[3] - 8, [15, 30, 31, 28], 6.5, rows, text_h=2.6)

    # --- условное обозначение
    legend_y = AREA[1] + 8
    add("layer.current", name=LAYER_PIPE)
    add("pline", vertices=[[AREA[2] + 8, legend_y], [AREA[2] + 28, legend_y]], width=0.8)
    add("layer.current", name=LAYER_TEXT)
    add("text", at=[AREA[2] + 32, legend_y], text=f"Труба {pipe_text}", height=2.6, justify="ML")
    add("text", at=[AREA[2] + 8, legend_y - 8], text="Условный план, не в масштабе диаметра", height=2.2, justify="ML")

    info = {
        "scale": scale,
        "total_m": geometry.total_m,
        "segments": len(spec.segments),
        "dimensions_expected": len(spec.segments) + _bend_count(spec),
    }
    return cmds, info


# ------------------------------ запуск CADCraft ------------------------------


def cli_path(cadcraft_dir: Path) -> Path:
    # абсолютный путь: программа запускается из временной папки
    return (Path(cadcraft_dir) / "cadcraft-cli").resolve()


def is_available(cadcraft_dir: Path) -> bool:
    return cli_path(cadcraft_dir).is_file()


def editor_dir(cadcraft_dir: Path) -> Path:
    return Path(cadcraft_dir) / "web"


def version(cadcraft_dir: Path) -> str | None:
    try:
        return (Path(cadcraft_dir) / "VERSION").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


@dataclass
class DrawingFiles:
    dxf: bytes
    dwg: bytes
    pdf: bytes
    png: bytes
    info: dict = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)


def render_preview(pdf: bytes, dpi: int = 110) -> bytes:
    """Картинка первой страницы PDF (белый лист, чёрные линии — как на печати)."""
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        return document[0].get_pixmap(dpi=dpi).tobytes("png")


_DXF_UNICODE_RE = re.compile(r"\\U\+([0-9A-Fa-f]{4})")


def _dxf_text(value: str) -> str:
    """Значение из DXF: в версиях R2000–R2004 не-ASCII символы записаны как \\U+0422."""
    return _DXF_UNICODE_RE.sub(lambda match: chr(int(match.group(1), 16)), value)


def read_route_polyline(dxf_text: str) -> list[tuple[float, float]]:
    """Вершины ломаной трассы, прочитанные из сохранённого DXF (слой PIPELINE, самая длинная ломаная).

    Нужны для независимой проверки: сверяем то, что реально записано в файл, с заказанным.
    """
    lines = dxf_text.splitlines()
    pairs = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]
    best: list[tuple[float, float]] = []
    index = 0
    while index < len(pairs):
        if pairs[index] == ("0", "LWPOLYLINE"):
            layer, vertices, pending_x = "", [], None
            index += 1
            while index < len(pairs) and pairs[index][0] != "0":
                code, value = pairs[index]
                if code == "8":
                    layer = _dxf_text(value)
                elif code == "10":
                    pending_x = float(value)
                elif code == "20" and pending_x is not None:
                    vertices.append((pending_x, float(value)))
                    pending_x = None
                index += 1
            if layer == LAYER_PIPE and len(vertices) > len(best):
                best = vertices
        else:
            index += 1
    return best


DWG_SIGNATURE = b"AC1015"  # DWG версии R2000: открывается почти любой программой


def verify_drawing(spec: RouteSpec, info: dict, cli_info: dict, dxf_text: str,
                   dwg_info: dict | None = None, dwg_bytes: bytes | None = None) -> list[dict]:
    """Сверка чертежа с заказом: размеры и узлы на месте, длины участков из файла совпадают с заказанными."""
    segments = len(spec.segments)
    counts = cli_info.get("counts", {})
    checks = [
        {"name": "Размеры участков и углов", "ok": counts.get("Dimension") == info["dimensions_expected"],
         "detail": f"в чертеже {counts.get('Dimension', 0)} из {info['dimensions_expected']}"},
        {"name": "Узловые точки", "ok": counts.get("Circle") == segments + 1,
         "detail": f"в чертеже {counts.get('Circle', 0)} из {segments + 1}"},
    ]
    extents = cli_info.get("extents") or [[0, 0], [0, 0]]
    inside = extents[0][0] >= -1e-6 and extents[0][1] >= -1e-6 and extents[1][0] <= SHEET_W + 1e-6 \
        and extents[1][1] <= SHEET_H + 1e-6
    checks.append({"name": "Всё в пределах листа A3", "ok": inside, "detail": "ничего не выходит за рамку листа"
                   if inside else f"выход за лист: {extents}"})

    if dwg_bytes is not None:
        same = bool(dwg_info) and dwg_info.get("counts") == cli_info.get("counts")
        signature = dwg_bytes.startswith(DWG_SIGNATURE)
        checks.append({"name": "DWG читается обратно", "ok": same and signature,
                       "detail": "формат R2000, объекты те же, что в DXF" if same and signature
                       else "DWG не совпал с DXF по составу объектов или формату"})

    vertices = read_route_polyline(dxf_text)
    scale = info["scale"]
    if len(vertices) == segments + 1:
        drawn = [math.hypot(b[0] - a[0], b[1] - a[1]) * scale / 1000 for a, b in zip(vertices, vertices[1:])]
        wanted = [seg.length_m for seg in spec.segments]
        worst = max(abs(d - w) / w for d, w in zip(drawn, wanted))
        ok = worst <= 0.001
        checks.append({"name": "Длины участков в файле = заказанным", "ok": ok,
                       "detail": f"итого {sum(drawn):.2f} м, расхождение не больше {worst * 100:.3f}%"
                       if ok else f"расхождение до {worst * 100:.2f}%"})
    else:
        checks.append({"name": "Длины участков в файле = заказанным", "ok": False,
                       "detail": f"в файле {len(vertices)} вершин трассы, ожидалось {segments + 1}"})
    return checks


def _cli(cli: Path, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(cli), *args], cwd=cwd, env={"PATH": "/usr/bin:/bin", "HOME": str(cwd)},
                          capture_output=True, text=True, timeout=CLI_TIMEOUT, check=False)


def draw_route(spec: RouteSpec, cadcraft_dir: Path) -> DrawingFiles:
    errors = validate_spec(spec)
    if errors:
        raise RouteSpecError("; ".join(errors))
    cli = cli_path(cadcraft_dir)
    if not cli.is_file():
        raise CadcraftUnavailable("CADCraft не установлен. Выполните scripts/fetch_cadcraft.sh и перезапустите сервер.")
    commands, info = build_commands(spec)
    with tempfile.TemporaryDirectory(prefix="route_") as tmp_name:
        tmp = Path(tmp_name)
        args = ["run", "--metric"]
        for command in commands:
            args += ["--cmd", command]
        # R2000 — самая совместимая версия (КОМПАС, nanoCAD, LibreCAD, AutoCAD): сохраняем DXF и DWG
        args += ["--cmd", 'saveas {"path": "route.dxf", "version": 2000}',
                 "--cmd", 'saveas {"path": "route.dwg", "version": 2000}']
        try:
            built = _cli(cli, args, tmp)
            if built.returncode != 0 or not (tmp / "route.dxf").is_file() or not (tmp / "route.dwg").is_file():
                raise CadcraftError("CADCraft не смог построить чертёж: " + (built.stderr or built.stdout).strip()[-300:])
            converted = _cli(cli, ["convert", "route.dxf", "route.pdf"], tmp)
            if converted.returncode != 0 or not (tmp / "route.pdf").is_file():
                raise CadcraftError("CADCraft не смог экспортировать PDF: "
                                    + (converted.stderr or converted.stdout).strip()[-200:])
            described = _cli(cli, ["info", "route.dxf"], tmp)
            cli_info = json.loads(described.stdout) if described.returncode == 0 else {}
            dwg_described = _cli(cli, ["info", "route.dwg"], tmp)
            dwg_info = json.loads(dwg_described.stdout) if dwg_described.returncode == 0 else {}
            dxf_bytes = (tmp / "route.dxf").read_bytes()
            dwg_bytes = (tmp / "route.dwg").read_bytes()
            checks = verify_drawing(spec, info, cli_info, dxf_bytes.decode("utf-8", errors="replace"),
                                    dwg_info, dwg_bytes)
            pdf_bytes = (tmp / "route.pdf").read_bytes()
            return DrawingFiles(dxf=dxf_bytes, dwg=dwg_bytes, pdf=pdf_bytes, png=render_preview(pdf_bytes),
                                info=info, checks=checks)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            raise CadcraftError("CADCraft не уложился по времени или вернул непонятный ответ") from exc


# ------------------------------ слова → схема трассы → чертёж ------------------------------

ROUTE_CAPABILITIES = [
    "трасса одной трубы из прямых участков с поворотами на заданные углы (до 12 участков)",
    "план на листе A3 в стандартном масштабе: размеры участков, углы, ведомость и основная надпись",
]

SYSTEM_PROMPT = """Ты помощник-проектировщик. Пользователь описывает трассу трубопровода на плане: \
участки с длинами и поворотами. Ты возвращаешь ТОЛЬКО один JSON-объект. Ничего кроме JSON.

Схема:
{
  "title": "название чертежа" (необязательно),
  "diameter_mm": наружный диаметр трубы в миллиметрах,
  "wall_mm": толщина стенки в миллиметрах (необязательно),
  "material": "сталь" (необязательно; если не назван, не указывай),
  "segments": [{"length_m": длина участка в МЕТРАХ, "turn_deg": поворот перед участком в градусах}, ...]
}

Правила.
- Трасса идёт от точки 1; первый участок направлен вправо, его turn_deg равен 0.
- turn_deg — поворот относительно прежнего направления: влево (налево, против часовой стрелки) — \
положительный, вправо (направо) — отрицательный. «Под прямым углом» без стороны — влево, 90.
- Если назван угол МЕЖДУ участками (внутренний), поворот равен 180 минус этот угол.
- Длины переведи в метры (1 км = 1000 м, 1 см = 0,01 м, 1 мм = 0,001 м). Диаметр — в миллиметрах: \
«Ø219×8» или «219х8» — это diameter_mm 219 и wall_mm 8.
- Не придумывай размеры. Если нет диаметра трубы или длин участков — верни {"clarify": "что именно уточнить"}.
- Поддерживается только одна труба из прямых участков с угловыми поворотами. Если просят ответвления, \
тройники, плавные отводы с радиусом, подъёмы и уклоны (продольный профиль), несколько ниток, привязку \
к координатам — верни {"unsupported": "что именно пока не поддерживается"}.
- Если основное нарисовать можно, а просят лишнее (глубина заложения, отметки, опоры, задвижки), нарисуй \
трассу без этого и перечисли пропущенное в "skipped": ["короткое название"] — без размеров и пояснений.

Примеры.
Описание: Труба Ø219×8, участок 12 м, затем налево на 90° и 8 м, затем направо на 45° и 10 м.
{"title": "План трассы", "diameter_mm": 219, "wall_mm": 8, "segments": [{"length_m": 12, "turn_deg": 0}, {"length_m": 8, "turn_deg": 90}, {"length_m": 10, "turn_deg": -45}]}

Описание: Трасса стальной трубы 114x6: 250 м, поворот направо на 60°, 120 м, поворот налево на 60°, 300 м.
{"title": "Трасса", "diameter_mm": 114, "wall_mm": 6, "material": "сталь", "segments": [{"length_m": 250, "turn_deg": 0}, {"length_m": 120, "turn_deg": -60}, {"length_m": 300, "turn_deg": 60}]}

Описание: Труба 159x6, два участка по 10 и 15 м под прямым углом, глубина заложения 1,5 м.
{"title": "Трасса", "diameter_mm": 159, "wall_mm": 6, "segments": [{"length_m": 10, "turn_deg": 0}, {"length_m": 15, "turn_deg": 90}], "skipped": ["глубина заложения"]}

Описание: Трасса с тройником и ответвлением на 15 м.
{"unsupported": "тройники и ответвления"}

Описание: Сделай чертёж трассы.
{"clarify": "Укажите диаметр трубы и длину каждого участка с поворотами между ними."}
"""


@dataclass
class Extraction:
    spec: RouteSpec | None = None
    clarify: str | None = None
    unsupported: str | None = None
    attempts: int = 0
    assumptions: list[str] = field(default_factory=list)


def extract_spec(description: str, llm) -> Extraction:
    """Описание → схема трассы. Ошибку проверки модель получает обратно и исправляется (до трёх попыток)."""
    description = description.strip()
    if not description:
        raise RouteSpecError("Опишите трассу: диаметр трубы, длины участков и повороты.")
    if len(description) > cad.MAX_DESCRIPTION:
        raise RouteSpecError(f"Описание слишком длинное (больше {cad.MAX_DESCRIPTION} символов).")

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Описание: {description}"}]
    errors: list[str] = []
    for attempt in range(1, cad.MAX_ATTEMPTS + 1):
        reply = llm.complete_text(messages, json_mode=True)
        spec = None
        try:
            data = cad._extract_json(reply)
            unsupported = data.get("unsupported")
            if isinstance(unsupported, str) and unsupported.strip():
                return Extraction(unsupported=unsupported.strip().rstrip("."), attempts=attempt)
            clarify = data.get("clarify")
            if isinstance(clarify, str) and clarify.strip():
                return Extraction(clarify=clarify.strip(), attempts=attempt)
            spec = RouteSpec.model_validate(data)
            errors = validate_spec(spec)
        except (ValueError, ValidationError) as exc:
            errors = cad._pydantic_errors(exc) if isinstance(exc, ValidationError) else [str(exc)]
        if spec is not None and not errors:
            assumptions: list[str] = []
            if not cad._MATERIAL_WORD.search(description):
                assumptions.append(f"Материал в описании не назван — принят: {spec.material}.")
            skipped = cad.clean_skipped_features(spec.skipped)
            if skipped:
                assumptions.append(cad.SKIPPED_NOTE_PREFIX + ", ".join(skipped) + ".")
            return Extraction(spec=spec, attempts=attempt, assumptions=assumptions)
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "Ошибки в схеме трассы: " + "; ".join(errors) +
                                                 ". Исправь и верни только JSON."}]
    raise RouteSpecError("Не удалось составить трассу по описанию: " + "; ".join(errors))


def describe(spec: RouteSpec, geometry: Geometry, scale: int | float) -> list[str]:
    """Как ассистент понял описание — строки для показа пользователю."""
    pipe = f"Ø{spec.diameter_mm:g}" + (f"×{spec.wall_mm:g}" if spec.wall_mm else "") + f", {spec.material}"
    lines = [f"Труба {pipe}", f"Трасса из {len(spec.segments)} уч., общая длина {_fmt(geometry.total_m)} м, масштаб {_scale_label(scale)}"]
    for number, segment in enumerate(spec.segments, start=1):
        if number == 1:
            turn = ""
        elif abs(segment.turn_deg) <= STRAIGHT_TOLERANCE_DEG:
            turn = ", продолжение по прямой"
        else:
            turn = f", поворот {'влево' if segment.turn_deg > 0 else 'вправо'} на {abs(segment.turn_deg):g}°"
        lines.append(f"Участок {number}: {_fmt(segment.length_m)} м{turn}")
    return lines


def build_result(spec: RouteSpec, cadcraft_dir: Path, *, assumptions: list[str], attempts: int) -> dict:
    files = draw_route(spec, cadcraft_dir)
    geometry = route_geometry(spec)
    return {
        "clarify": None,
        "title": spec.title,
        "summary": describe(spec, geometry, files.info["scale"]),
        "assumptions": assumptions,
        "attempts": attempts,
        "spec": spec.model_dump(exclude_none=True, exclude={"skipped"}),
        "checks": files.checks,
        "metrics": {"total_m": round(geometry.total_m, 2), "scale": f"1:{files.info['scale']:g}",
                    "segments": len(spec.segments)},
        "images": {"preview": base64.b64encode(files.png).decode("ascii")},
        "pdf_base64": base64.b64encode(files.pdf).decode("ascii"),
        "dxf_base64": base64.b64encode(files.dxf).decode("ascii"),
        "dwg_base64": base64.b64encode(files.dwg).decode("ascii"),
    }


def generate(description: str, llm, cadcraft_dir: Path) -> dict:
    """Описание → чертёж трассы. Вопрос на уточнение приходит в `clarify`, отказ — в `unsupported`."""
    if not is_available(cadcraft_dir):
        raise CadcraftUnavailable("CADCraft не установлен. Выполните scripts/fetch_cadcraft.sh и перезапустите сервер.")
    extraction = extract_spec(description, llm)
    if extraction.unsupported:
        return {"clarify": None, "unsupported": extraction.unsupported, "can_build": ROUTE_CAPABILITIES,
                "attempts": extraction.attempts}
    if extraction.clarify:
        return {"clarify": extraction.clarify, "attempts": extraction.attempts}
    assert extraction.spec is not None
    return build_result(extraction.spec, cadcraft_dir, assumptions=extraction.assumptions,
                        attempts=extraction.attempts)


def rebuild(spec_data: dict, cadcraft_dir: Path, *, notes: list[str]) -> dict:
    """Повторное построение по сохранённой схеме — без обращения к модели."""
    if not is_available(cadcraft_dir):
        raise CadcraftUnavailable("CADCraft не установлен. Выполните scripts/fetch_cadcraft.sh и перезапустите сервер.")
    try:
        spec = RouteSpec.model_validate(spec_data)
    except ValidationError as exc:
        raise RouteSpecError("Сохранённая схема повреждена: " + "; ".join(cad._pydantic_errors(exc))) from exc
    errors = validate_spec(spec)
    if errors:
        raise RouteSpecError("Сохранённая схема больше не проходит проверку: " + "; ".join(errors))
    return build_result(spec, cadcraft_dir, assumptions=notes, attempts=1)
