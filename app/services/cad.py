"""Конструктор деталей: описание словами → проверенная спецификация → SolidCraft → модель.

Принцип: языковая модель НЕ пишет команды CAD. Она заполняет небольшую схему (спецификацию
детали: форма, размеры, отверстия), а команды для SolidCraft строит наш код. Отсюда:

1. модель не может выдать произвольную команду (запись файлов, системные вызовы) — в CAD
   попадает только то, что собрал код из проверенных полей;
2. ошибки модели (отверстие за краем детали, пропавший размер) ловятся проверкой до запуска
   CAD, а модель получает текст ошибки и исправляется;
3. результат сверяется с расчётом: габариты и объём модели сравниваются с аналитическими.
   Это проверка и самой модели, и построения;
4. недостающие размеры не додумываются: модель возвращает вопрос, и он доходит до пользователя.

SolidCraft — сторонний открытый проект (MIT/Apache-2.0, github.com/bherbruck/solidcraft),
запускается как отдельная программа `solidcraft-cli`; наш вклад — слой «слова → спецификация →
команды» и проверки. Ограничение: поддерживаются простые детали (плита/блок и цилиндр с
отверстиями), для сложной геометрии нужен настоящий CAD.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

MIN_DIM = 0.5  # мм
MAX_DIM = 2000.0  # мм
MAX_HOLES = 100
MAX_DESCRIPTION = 1000
CLI_TIMEOUT = 30  # секунд на один запуск
MAX_ATTEMPTS = 3  # первая попытка + две правки по тексту ошибки

MIN_PIPE_ANGLE = 10.0  # градусов между осями труб
MAX_PIPE_ANGLE = 170.0
_PIPE_SAMPLES = 3600  # точек на окружности при расчёте габаритов стыка

BBOX_TOLERANCE_MM = 0.05
VOLUME_TOLERANCE = 0.003  # 0.3%

DEFAULT_MATERIAL = "Steel"
MATERIALS_RU = {
    "Steel": "сталь",
    "Stainless Steel": "нержавеющая сталь",
    "Aluminum": "алюминий",
    "Brass": "латунь",
    "Copper": "медь",
    "Titanium": "титан",
    "Cast Iron": "чугун",
    "ABS Plastic": "пластик ABS",
    "PLA": "пластик PLA",
    "Nylon": "нейлон",
    "Polycarbonate": "поликарбонат",
}


# Что умеет конструктор сейчас — показывается пользователю вместе с «пока не поддерживается».
CAPABILITIES = [
    "плита или блок со сквозными отверстиями (по углам, сеткой, в заданных точках), скруглённые углы",
    "цилиндр, диск, фланец, втулка, прямая труба (в том числе с отверстиями по окружности)",
    "две соединённые трубы под углом",
]


class CadError(Exception):
    """Ошибка конструктора с текстом, который можно показать пользователю."""


class CadUnavailable(CadError):
    """SolidCraft не установлен (scripts/fetch_solidcraft.sh)."""


class CadSpecError(CadError):
    """Модель не смогла дать корректную спецификацию детали."""


# ------------------------------ спецификация ------------------------------


class Hole(BaseModel):
    model_config = ConfigDict(extra="ignore")
    x: float
    y: float
    diameter: float


class HoleGrid(BaseModel):
    """Сетка сквозных отверстий на плите: равномерно между отступами от краёв."""

    model_config = ConfigDict(extra="ignore")
    count_x: int = Field(default=1, ge=1)
    count_y: int = Field(default=1, ge=1)
    offset_x: float = Field(default=0, ge=0)  # от края до центра крайнего отверстия
    offset_y: float = Field(default=0, ge=0)
    diameter: float


class BoltCircle(BaseModel):
    """Отверстия, равномерно расставленные по окружности (фланец)."""

    model_config = ConfigDict(extra="ignore")
    count: int = Field(ge=1)
    circle_diameter: float
    diameter: float
    start_angle_deg: float = 0


# Ключи ответа модели вне схемы, о которых не стоит сообщать пользователю.
_HARMLESS_EXTRA = {"description", "name", "notes", "note", "comment", "comments"}


class PartSpec(BaseModel):
    # Лишние поля не отбрасываем молча: о них сообщаем пользователю (см. unknown_fields).
    model_config = ConfigDict(extra="allow")
    title: str = "Деталь"
    material: str = DEFAULT_MATERIAL
    shape: Literal["box", "cylinder", "pipes"]
    # box: length (X), width (Y), height (Z); cylinder: diameter, height; все размеры в мм
    # pipes — две трубы под углом: diameter (наружный), wall_thickness, length и length2
    # (длины от точки пересечения осей), angle_deg (угол между трубами)
    length: float | None = None
    width: float | None = None
    height: float | None = None
    diameter: float | None = None
    wall_thickness: float | None = None  # труба: толщина стенки (у цилиндра задаёт центральное отверстие)
    length2: float | None = None
    angle_deg: float | None = None
    bore_diameter: float | None = None  # цилиндр: центральное сквозное отверстие
    corner_radius: float | None = None  # box: скругление вертикальных рёбер
    holes: list[Hole] = Field(default_factory=list)
    hole_grid: HoleGrid | None = None
    bolt_circle: BoltCircle | None = None
    # Что из описания схема не умеет (фаски, резьба…): деталь строится без этого, а пользователь
    # видит список «пока не поддерживается».
    skipped: list[str] = Field(default_factory=list)


def unknown_fields(spec: PartSpec) -> list[str]:
    """Поля, которые модель вернула, но схема их не знает (значит, в модель они не попали)."""
    return sorted(k for k in (spec.model_extra or {}) if k.lower() not in _HARMLESS_EXTRA)


def effective_bore(spec: PartSpec) -> float | None:
    """Диаметр центрального отверстия цилиндра: задан прямо или через толщину стенки."""
    if spec.bore_diameter:
        return spec.bore_diameter
    if spec.wall_thickness and spec.diameter:
        return spec.diameter - 2 * spec.wall_thickness
    return None


def pipe_leg_lengths(spec: PartSpec) -> tuple[float, float]:
    first = float(spec.length)  # type: ignore[arg-type]
    return first, float(spec.length2 if spec.length2 else first)


def min_pipe_leg(spec: PartSpec) -> float:
    """Наименьшая длина участка, при которой косой стык целиком помещается на трубе."""
    radius = float(spec.diameter) / 2  # type: ignore[arg-type]
    return radius / math.tan(math.radians(float(spec.angle_deg)) / 2) + radius  # type: ignore[arg-type]


def _dims_error(name: str, value: float | None) -> str | None:
    if value is None:
        return f"не указан размер «{name}»"
    if not (MIN_DIM <= value <= MAX_DIM):
        return f"размер «{name}» = {value:g} мм вне допустимого диапазона {MIN_DIM:g}–{MAX_DIM:g} мм"
    return None


def hole_positions(spec: PartSpec) -> list[tuple[float, float, float]]:
    """Все сквозные отверстия (x, y, диаметр) в системе координат детали.

    Плита: начало координат в углу, x вдоль длины, y вдоль ширины. Цилиндр: начало в центре.
    """
    if spec.shape == "pipes":
        return []
    result = [(h.x, h.y, h.diameter) for h in spec.holes]
    grid = spec.hole_grid
    if grid and spec.shape == "box" and spec.length and spec.width:
        def axis(count: int, offset: float, size: float) -> list[float]:
            if count == 1:
                return [size / 2]
            step = (size - 2 * offset) / (count - 1)
            return [offset + i * step for i in range(count)]

        for x in axis(grid.count_x, grid.offset_x, spec.length):
            for y in axis(grid.count_y, grid.offset_y, spec.width):
                result.append((x, y, grid.diameter))
    circle = spec.bolt_circle
    if circle and spec.shape == "cylinder":
        radius = circle.circle_diameter / 2
        for k in range(circle.count):
            angle = math.radians(circle.start_angle_deg) + 2 * math.pi * k / circle.count
            result.append((radius * math.cos(angle), radius * math.sin(angle), circle.diameter))
    return result


def validate_spec(spec: PartSpec) -> list[str]:
    """Проверка спецификации. Пустой список — можно строить. Тексты ошибок понятны модели и человеку."""
    errors: list[str] = []
    pipe_only = [n for n in ("wall_thickness", "length2", "angle_deg") if getattr(spec, n) is not None]
    if spec.shape == "box":
        errors += [e for e in (_dims_error("длина", spec.length), _dims_error("ширина", spec.width),
                               _dims_error("высота (толщина)", spec.height)) if e]
        if spec.diameter or spec.bore_diameter or spec.bolt_circle or pipe_only:
            errors.append("для shape=box нельзя указывать diameter, bore_diameter, bolt_circle, "
                          "wall_thickness, length2 и angle_deg")
        if spec.corner_radius is not None and not errors:
            limit = min(spec.length, spec.width) / 2  # type: ignore[type-var,operator]
            if spec.corner_radius <= 0 or spec.corner_radius >= limit:
                errors.append(f"радиус скругления углов должен быть больше 0 и меньше {limit:g} мм")
    elif spec.shape == "cylinder":
        errors += [e for e in (_dims_error("диаметр", spec.diameter), _dims_error("высота", spec.height)) if e]
        if spec.length or spec.width or spec.corner_radius or spec.hole_grid or spec.length2 or spec.angle_deg:
            errors.append("для shape=cylinder нельзя указывать length, width, corner_radius, hole_grid, "
                          "length2 и angle_deg")
        if spec.diameter and not errors:
            bore = effective_bore(spec)
            if spec.bore_diameter and spec.wall_thickness and abs(
                spec.diameter - 2 * spec.wall_thickness - spec.bore_diameter
            ) > 0.01:
                errors.append("bore_diameter и wall_thickness противоречат друг другу: укажите что-то одно")
            elif bore is not None and not (0 < bore < spec.diameter):
                errors.append("диаметр центрального отверстия должен быть больше 0 и меньше диаметра детали")
    else:  # pipes
        errors += [e for e in (_dims_error("наружный диаметр", spec.diameter),
                               _dims_error("толщина стенки", spec.wall_thickness),
                               _dims_error("длина первой трубы", spec.length)) if e]
        if spec.length2 is not None:
            error = _dims_error("длина второй трубы", spec.length2)
            errors += [error] if error else []
        if spec.angle_deg is None:
            errors.append("не указан угол между трубами (angle_deg)")
        elif not (MIN_PIPE_ANGLE <= spec.angle_deg <= MAX_PIPE_ANGLE):
            errors.append(f"угол между трубами должен быть от {MIN_PIPE_ANGLE:g}° до {MAX_PIPE_ANGLE:g}°")
        if spec.width or spec.height or spec.bore_diameter or spec.corner_radius or spec.holes \
                or spec.hole_grid or spec.bolt_circle:
            errors.append("для shape=pipes нельзя указывать width, height, bore_diameter, corner_radius, "
                          "holes, hole_grid и bolt_circle")
        if not errors:
            if spec.wall_thickness >= spec.diameter / 2:  # type: ignore[operator]
                errors.append("толщина стенки должна быть меньше радиуса трубы")
            else:
                needed = min_pipe_leg(spec)
                for number, length in enumerate(pipe_leg_lengths(spec), start=1):
                    if length < needed:
                        errors.append(
                            f"труба {number} слишком короткая для косого стыка под углом {spec.angle_deg:g}°: "
                            f"нужно не меньше {needed:.0f} мм, указано {length:g} мм"
                        )

    if errors:
        return errors

    holes = hole_positions(spec)
    if len(holes) > MAX_HOLES:
        return [f"слишком много отверстий ({len(holes)}), допустимо не более {MAX_HOLES}"]
    for n, (x, y, d) in enumerate(holes, start=1):
        if not (MIN_DIM <= d <= MAX_DIM):
            errors.append(f"отверстие {n}: диаметр {d:g} мм вне допустимого диапазона")
            continue
        if spec.shape == "box":
            if x - d / 2 < -1e-6 or y - d / 2 < -1e-6 or x + d / 2 > spec.length + 1e-6 or y + d / 2 > spec.width + 1e-6:  # type: ignore[operator]
                errors.append(
                    f"отверстие {n} (x={x:g}, y={y:g}, Ø{d:g}) выходит за границы детали "
                    f"{spec.length:g}×{spec.width:g} мм"
                )
        else:
            radius = spec.diameter / 2  # type: ignore[operator]
            distance = math.hypot(x, y)
            if distance + d / 2 > radius + 1e-6:
                errors.append(f"отверстие {n} (Ø{d:g}) выходит за границу детали Ø{spec.diameter:g} мм")
            elif effective_bore(spec) and distance - d / 2 < effective_bore(spec) / 2 - 1e-6:  # type: ignore[operator]
                errors.append(f"отверстие {n} (Ø{d:g}) пересекается с центральным отверстием")
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            (x1, y1, d1), (x2, y2, d2) = holes[i], holes[j]
            if math.hypot(x1 - x2, y1 - y2) < (d1 + d2) / 2 - 1e-6:
                errors.append(f"отверстия {i + 1} и {j + 1} пересекаются")
                if len(errors) >= 5:
                    return errors
    return errors


def normalize_material(value: str) -> tuple[str, bool]:
    """Название материала из библиотеки SolidCraft и признак «подставлен по умолчанию»."""
    if value in MATERIALS_RU:
        return value, False
    lowered = value.strip().lower()
    for english, russian in MATERIALS_RU.items():
        if lowered in (english.lower(), russian):
            return english, False
    return DEFAULT_MATERIAL, True


# ------------------------------ команды CAD ------------------------------


def _pipe_path_points(spec: PartSpec) -> list[list[float]]:
    """Ось двух труб в плоскости XY: первая идёт вдоль +X от точки пересечения осей, вторая — под углом."""
    first, second = pipe_leg_lengths(spec)
    angle = math.radians(float(spec.angle_deg))  # type: ignore[arg-type]
    return [[first, 0], [0, 0], [round(second * math.cos(angle), 6), round(second * math.sin(angle), 6)]]


def spec_to_commands(spec: PartSpec) -> list[dict]:
    """Список команд SolidCraft по проверенной спецификации. Строка из модели в команды не попадает,
    кроме материала из фиксированного списка."""
    material, _ = normalize_material(spec.material)
    commands: list[dict] = []
    if spec.shape == "pipes":
        commands += [
            {"command": "sketch.create", "params": {"plane": "XY", "name": "Path"}},
            {"command": "sketch.line", "params": {"points": _pipe_path_points(spec), "ids": ["l1", "l2"]}},
            {"command": "sketch.finish", "params": {}},
            {"command": "solid.pipe", "params": {
                "path_sketch": "Path", "path": ["l1", "l2"], "diameter": float(spec.diameter),  # type: ignore[arg-type]
                "wall": float(spec.wall_thickness), "body_name": "Part"}},  # type: ignore[arg-type]
        ]
        commands.append({"command": "material.assign", "params": {"bodies": ["Part"], "material": material}})
        return commands
    height = float(spec.height)  # type: ignore[arg-type]
    if spec.shape == "box":
        length, width = float(spec.length), float(spec.width)  # type: ignore[arg-type]
        commands.append({"command": "solid.box", "params": {
            "length": length, "width": width, "height": height, "corner": [0, 0, 0], "body_name": "Part"}})
        if spec.corner_radius:
            edges = [[0, 0, height / 2], [length, 0, height / 2], [length, width, height / 2], [0, width, height / 2]]
            commands.append({"command": "solid.fillet", "params": {"edges": edges, "radius": spec.corner_radius}})
    else:
        commands.append({"command": "solid.cylinder", "params": {
            "diameter": float(spec.diameter), "height": height, "base": [0, 0, 0],  # type: ignore[arg-type]
            "axis": "Z", "body_name": "Part"}})
        bore = effective_bore(spec)
        if bore:
            commands.append({"command": "solid.cylinder", "params": {
                "diameter": bore, "height": height, "base": [0, 0, 0], "axis": "Z",
                "operation": "cut", "targets": ["Part"]}})
    for x, y, d in hole_positions(spec):
        commands.append({"command": "solid.hole", "params": {"position": [round(x, 4), round(y, 4), height], "diameter": d}})
    commands.append({"command": "material.assign", "params": {"bodies": ["Part"], "material": material}})
    return commands


def _pipe_joint_size(spec: PartSpec) -> list[float]:
    """Габариты стыка двух труб (X, Y, Z по возрастанию): дальние торцы и эллипсы косого среза."""
    outer = float(spec.diameter) / 2  # type: ignore[arg-type]
    angle = math.radians(float(spec.angle_deg))  # type: ignore[arg-type]
    first, second = pipe_leg_lengths(spec)
    axes = [(1.0, 0.0), (math.cos(angle), math.sin(angle))]
    # нормаль плоскости среза — биссектриса между осями; у каждой трубы она смотрит в её сторону
    nx, ny = axes[0][0] - axes[1][0], axes[0][1] - axes[1][1]
    norm = math.hypot(nx, ny)
    normals = [(nx / norm, ny / norm), (-nx / norm, -ny / norm)]
    points: list[tuple[float, float, float]] = []
    for (ax, ay), (mx, my), length in zip(axes, normals, (first, second)):
        across = (-ay, ax)  # в плоскости XY, перпендикулярно оси; третья компонента — Z
        slope = mx * ax + my * ay  # cos угла между осью и нормалью среза
        for k in range(_PIPE_SAMPLES):
            phi = 2 * math.pi * k / _PIPE_SAMPLES
            cx, cy, cz = (outer * math.cos(phi) * across[0], outer * math.cos(phi) * across[1], outer * math.sin(phi))
            points.append((length * ax + cx, length * ay + cy, cz))  # дальний торец
            shift = -(mx * cx + my * cy) / slope  # сдвиг вдоль оси до плоскости среза
            points.append((cx + shift * ax, cy + shift * ay, cz))
    return sorted(max(p[i] for p in points) - min(p[i] for p in points) for i in range(3))


def expected_geometry(spec: PartSpec) -> dict:
    """Габариты и объём по формулам — для сверки с тем, что построил CAD."""
    if spec.shape == "pipes":
        outer = float(spec.diameter) / 2  # type: ignore[arg-type]
        inner = outer - float(spec.wall_thickness)  # type: ignore[arg-type]
        # стык со срезом по биссектрисе: объём равен объёму прямой трубы суммарной длины по осям
        return {"size": _pipe_joint_size(spec),
                "volume": math.pi * (outer**2 - inner**2) * sum(pipe_leg_lengths(spec))}
    height = float(spec.height)  # type: ignore[arg-type]
    holes_volume = sum(math.pi * d * d / 4 * height for _, _, d in hole_positions(spec))
    if spec.shape == "box":
        length, width = float(spec.length), float(spec.width)  # type: ignore[arg-type]
        r = spec.corner_radius or 0.0
        corners = 4 * (1 - math.pi / 4) * r * r * height
        return {"size": sorted([length, width, height]), "volume": length * width * height - corners - holes_volume}
    diameter = float(spec.diameter)  # type: ignore[arg-type]
    bore = effective_bore(spec) or 0.0
    return {"size": sorted([diameter, diameter, height]),
            "volume": math.pi * (diameter**2 - bore**2) / 4 * height - holes_volume}


def describe(spec: PartSpec) -> list[str]:
    """Как ассистент понял описание — строки для показа пользователю."""
    lines: list[str] = []
    if spec.shape == "pipes":
        first, second = pipe_leg_lengths(spec)
        lines.append(f"Две соединённые трубы Ø{spec.diameter:g} мм, стенка {spec.wall_thickness:g} мм")
        lines.append(f"Угол между трубами {spec.angle_deg:g}°, длины от точки пересечения осей: "
                     f"{first:g} и {second:g} мм")
        lines.append("Стык — косой срез по биссектрисе")
    elif spec.shape == "box":
        lines.append(f"Плита/блок {spec.length:g} × {spec.width:g} × {spec.height:g} мм")
        if spec.corner_radius:
            lines.append(f"Скругление углов R{spec.corner_radius:g} мм")
    else:
        lines.append(f"Цилиндр Ø{spec.diameter:g} мм, высота {spec.height:g} мм")
        bore = effective_bore(spec)
        if bore:
            lines.append(f"Центральное сквозное отверстие Ø{bore:g} мм")
    holes = hole_positions(spec)
    if holes:
        by_diameter: dict[float, int] = {}
        for _, _, d in holes:
            by_diameter[d] = by_diameter.get(d, 0) + 1
        lines.append("Сквозные отверстия: " + ", ".join(f"{n} × Ø{d:g} мм" for d, n in sorted(by_diameter.items())))
    material, _ = normalize_material(spec.material)
    lines.append(f"Материал: {MATERIALS_RU[material]}")
    return lines


# ------------------------------ запуск SolidCraft ------------------------------


@dataclass
class CadRun:
    step: bytes
    images: dict[str, bytes]
    body: dict  # результат `solidcraft-cli eval` для детали: bbox, volume_mm3, mass_g, material


def cli_path(solidcraft_dir: Path) -> Path:
    return solidcraft_dir / "solidcraft-cli"


def editor_dir(solidcraft_dir: Path) -> Path:
    return solidcraft_dir / "web"


def is_available(solidcraft_dir: Path) -> bool:
    return cli_path(solidcraft_dir).is_file()


def version(solidcraft_dir: Path) -> str | None:
    try:
        return (solidcraft_dir / "VERSION").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _cli(cli: Path, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Запуск CLI в пустой временной папке, без переменных окружения пользователя."""
    return subprocess.run(
        [str(cli), *args],
        cwd=cwd,
        env={"PATH": "/usr/bin:/bin", "HOME": str(cwd)},
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT,
        check=False,
    )


def run_cad(commands: list[dict], solidcraft_dir: Path) -> CadRun:
    cli = cli_path(solidcraft_dir)
    if not cli.is_file():
        raise CadUnavailable(
            "SolidCraft не установлен. Выполните scripts/fetch_solidcraft.sh и перезапустите сервер."
        )
    with tempfile.TemporaryDirectory(prefix="cad_") as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "script.json").write_text(json.dumps({"commands": commands}), encoding="utf-8")
        try:
            built = _cli(cli, ["run", "script.json", "--out", "part.step", "--quiet"], tmp)
            if built.returncode != 0 or not (tmp / "part.step").is_file():
                raise CadError("CAD не смог построить деталь: " + (built.stderr or built.stdout).strip()[:300])
            evaluated = _cli(cli, ["eval", "script.json"], tmp)
            bodies = json.loads(evaluated.stdout).get("bodies") or []
            if evaluated.returncode != 0 or not bodies:
                raise CadError("CAD построил пустой результат (тела нет)")
            images: dict[str, bytes] = {}
            for view in ("iso", "top"):
                shot = _cli(cli, ["snapshot", "script.json", "--out", f"{view}.png",
                                  "--width", "800", "--height", "600", "--view", view], tmp)
                if shot.returncode == 0 and (tmp / f"{view}.png").is_file():
                    images[view] = (tmp / f"{view}.png").read_bytes()
            return CadRun(step=(tmp / "part.step").read_bytes(), images=images, body=bodies[0])
        except subprocess.TimeoutExpired as exc:
            raise CadError(f"CAD не уложился в {CLI_TIMEOUT} с") from exc
        except (json.JSONDecodeError, KeyError) as exc:
            raise CadError("CAD вернул непонятный ответ") from exc


def verify(spec: PartSpec, body: dict) -> list[dict]:
    """Сверка построенной модели с расчётом: габариты и объём."""
    expected = expected_geometry(spec)
    low, high = body["bbox"]["min"], body["bbox"]["max"]
    actual_size = sorted(abs(h - l) for l, h in zip(low, high))
    size_ok = all(abs(a - e) <= BBOX_TOLERANCE_MM for a, e in zip(actual_size, expected["size"]))
    volume = float(body["volume_mm3"])
    volume_ok = abs(volume - expected["volume"]) <= VOLUME_TOLERANCE * expected["volume"]
    return [
        {"name": "Габариты", "ok": size_ok,
         "detail": "совпали с заданными" if size_ok else
                   f"ожидались {expected['size']}, построено {[round(v, 2) for v in actual_size]}"},
        {"name": "Объём", "ok": volume_ok,
         "detail": f"{volume / 1000:.2f} см³ совпал с расчётом" if volume_ok else
                   f"расчёт {expected['volume'] / 1000:.2f} см³, построено {volume / 1000:.2f} см³"},
    ]


# ------------------------------ модель → спецификация ------------------------------

SYSTEM_PROMPT = """Ты помощник-конструктор. Пользователь описывает простую деталь словами. \
Ты возвращаешь ТОЛЬКО один JSON-объект — спецификацию детали. Ничего кроме JSON.

Все размеры в миллиметрах (1 см = 10 мм, 1 м = 1000 мм). Ставь только размеры, которые названы \
в описании. Если размеров не хватает (нет длины, ширины, толщины или диаметра), НЕ придумывай их, \
а верни {"clarify": "что именно уточнить"} на русском языке.

Схема:
{
  "title": "короткое название детали",
  "material": "Steel | Stainless Steel | Aluminum | Brass | Copper | Titanium | Cast Iron | ABS Plastic | PLA | Nylon | Polycarbonate",
  "shape": "box" (плита, пластина, брусок), "cylinder" (диск, фланец, втулка, шайба, одна прямая труба) \
или "pipes" (две соединённые трубы под углом),
  "length": X-размер, "width": Y-размер, "height": толщина/высота (только для box; height есть и у cylinder),
  "diameter": диаметр (cylinder, а для pipes — наружный диаметр труб),
  "bore_diameter": диаметр центрального сквозного отверстия (только для cylinder, необязательно),
  "wall_thickness": толщина стенки трубы (pipes — обязательно; для cylinder можно вместо bore_diameter),
  для pipes: "length" — длина первой трубы, "length2" — длина второй (если не названа, равна первой), обе от точки \
пересечения осей; "angle_deg" — угол между трубами в градусах (90 — прямой угол, 60 — острый, 180 — прямая труба),
  "corner_radius": радиус скругления углов (только для box, необязательно),
  "holes": [{"x": от края вдоль длины, "y": от края вдоль ширины, "diameter": Ø}]  (необязательно; для cylinder x и y отсчитываются от центра),
  "hole_grid": {"count_x": N, "count_y": M, "offset_x": отступ от края до центра отверстия, "offset_y": ..., "diameter": Ø}  (только для box: сетка отверстий; 4 отверстия по углам = count_x 2, count_y 2),
  "bolt_circle": {"count": N, "circle_diameter": диаметр окружности центров, "diameter": Ø отверстия}  (только для cylinder: отверстия по окружности)
}
Не добавляй поля, которых нет в схеме. Материал, если не назван, не указывай.

Поддерживаются только плита/брусок, цилиндр/труба и две трубы под углом. Если просят другой предмет \
(тройник, ответвление, уголок и кронштейн, болт, шестерня, труба с плавным изгибом, сборка из нескольких \
деталей, шар, профиль) — верни {"unsupported": "что именно пока не поддерживается"}. Если предмет \
поддерживается, а просят лишнее для схемы (фаски, резьба, пазы, надписи) — построй деталь без этого и \
перечисли пропущенное в поле "skipped": ["что не учтено"]. Никогда не подменяй один предмет другим: \
уголок или кронштейн — это не плита, это "unsupported".

Примеры.
Описание: Стальная пластина 100 на 60 мм, толщина 5 мм, четыре отверстия диаметром 6 мм по углам на расстоянии 10 мм от краёв, углы скруглить радиусом 5.
{"title": "Монтажная пластина", "material": "Steel", "shape": "box", "length": 100, "width": 60, "height": 5, "corner_radius": 5, "hole_grid": {"count_x": 2, "count_y": 2, "offset_x": 10, "offset_y": 10, "diameter": 6}}

Описание: Фланец из нержавейки, диаметр 150, толщина 12, центральное отверстие 60, шесть отверстий по 14 на окружности диаметром 110.
{"title": "Фланец", "material": "Stainless Steel", "shape": "cylinder", "diameter": 150, "height": 12, "bore_diameter": 60, "bolt_circle": {"count": 6, "circle_diameter": 110, "diameter": 14}}

Описание: Алюминиевый брусок 20 см на 5 см, высота 1 см, два отверстия диаметром 8: на 50 и на 150 мм от левого края, по центру ширины.
{"title": "Брусок", "material": "Aluminum", "shape": "box", "length": 200, "width": 50, "height": 10, "holes": [{"x": 50, "y": 25, "diameter": 8}, {"x": 150, "y": 25, "diameter": 8}]}

Описание: Две соединённые трубы под углом 60 градусов: наружный диаметр 100 мм, толщина стенки 5 мм, длина каждой 500 мм.
{"title": "Стык труб", "material": "Steel", "shape": "pipes", "diameter": 100, "wall_thickness": 5, "length": 500, "length2": 500, "angle_deg": 60}

Описание: Стальная труба наружным диаметром 60 мм со стенкой 4 мм, длина 1 м.
{"title": "Труба", "material": "Steel", "shape": "cylinder", "diameter": 60, "wall_thickness": 4, "height": 1000}

Описание: Тройник из труб диаметром 100 мм.
{"unsupported": "тройники и ответвления труб"}

Описание: Кронштейн уголком 50 на 50 мм, толщина 3, длина 100.
{"unsupported": "уголки и кронштейны (детали из нескольких плоскостей)"}

Описание: Стальная пластина 100 на 60, толщина 5, фаски 2 мм по кромкам.
{"title": "Пластина", "material": "Steel", "shape": "box", "length": 100, "width": 60, "height": 5, "skipped": ["фаски по кромкам"]}

Описание: Сделай втулку.
{"clarify": "Укажите наружный диаметр, внутренний диаметр и высоту втулки в миллиметрах."}
"""


_MATERIAL_WORD = re.compile(
    r"сталь|стальн|нержав|алюмин|латун|медн|\bмед[иь]\b|титан|чугун|пластик|abs|pla\b|нейлон|поликарбонат",
    re.IGNORECASE,
)


@dataclass
class Generation:
    spec: PartSpec | None = None
    clarify: str | None = None
    unsupported: str | None = None
    attempts: int = 0
    assumptions: list[str] = field(default_factory=list)


def _extract_json(text: str) -> dict:
    """Первый JSON-объект из ответа модели (она может обернуть его в ```json)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("в ответе нет JSON-объекта")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(obj, dict):
        raise ValueError("ответ — не JSON-объект")
    return obj


def _pydantic_errors(exc: ValidationError) -> list[str]:
    return [f"поле «{'.'.join(str(p) for p in e['loc'])}»: {e['msg']}" for e in exc.errors()[:5]]


def extract_spec(description: str, llm) -> Generation:
    """Описание → спецификация. Ошибку проверки модель получает обратно и исправляется (до MAX_ATTEMPTS)."""
    description = description.strip()
    if not description:
        raise CadSpecError("Опишите деталь: форму, размеры и отверстия.")
    if len(description) > MAX_DESCRIPTION:
        raise CadSpecError(f"Описание слишком длинное (больше {MAX_DESCRIPTION} символов).")

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Описание: {description}"}]
    errors: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = llm.complete_text(messages, json_mode=True)
        try:
            data = _extract_json(reply)
            unsupported = data.get("unsupported")
            if isinstance(unsupported, str) and unsupported.strip():
                return Generation(unsupported=unsupported.strip().rstrip("."), attempts=attempt)
            clarify = data.get("clarify")
            if isinstance(clarify, str) and clarify.strip():
                return Generation(clarify=clarify.strip(), attempts=attempt)
            spec = PartSpec.model_validate(data)
            errors = validate_spec(spec)
        except (ValueError, ValidationError) as exc:
            errors = _pydantic_errors(exc) if isinstance(exc, ValidationError) else [str(exc)]
            spec = None
        if spec is not None and not errors:
            assumptions = []
            if not _MATERIAL_WORD.search(description):
                material = MATERIALS_RU[normalize_material(spec.material)[0]]
                assumptions.append(f"Материал в описании не назван — принят: {material}.")
            skipped = [str(item).strip() for item in spec.skipped if str(item).strip()]
            if skipped:
                assumptions.append("Пока не поддерживается и не учтено: " + ", ".join(skipped) + ".")
            ignored = unknown_fields(spec)
            if ignored:
                assumptions.append("Не учтено (такого нет в схеме детали): " + ", ".join(ignored) + ".")
            return Generation(spec=spec, attempts=attempt, assumptions=assumptions)
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "Ошибки в спецификации: " + "; ".join(errors) +
                                        ". Исправь и верни только JSON."},
        ]
    raise CadSpecError("Не удалось составить корректную деталь по описанию: " + "; ".join(errors))


# ------------------------------ всё вместе ------------------------------


def generate(description: str, llm, solidcraft_dir: Path) -> dict:
    """Описание → модель. Возвращает словарь для API; вопрос на уточнение приходит в `clarify`."""
    if not is_available(solidcraft_dir):
        raise CadUnavailable("SolidCraft не установлен. Выполните scripts/fetch_solidcraft.sh и перезапустите сервер.")
    generation = extract_spec(description, llm)
    if generation.unsupported:
        return {"clarify": None, "unsupported": generation.unsupported, "can_build": CAPABILITIES,
                "attempts": generation.attempts}
    if generation.clarify:
        return {"clarify": generation.clarify, "attempts": generation.attempts}

    spec = generation.spec
    assert spec is not None
    commands = spec_to_commands(spec)
    run = run_cad(commands, solidcraft_dir)
    body = run.body
    low, high = body["bbox"]["min"], body["bbox"]["max"]
    return {
        "clarify": None,
        "title": spec.title,
        "summary": describe(spec),
        "assumptions": generation.assumptions,
        "attempts": generation.attempts,
        "spec": spec.model_dump(exclude_none=True, exclude={"skipped"}),
        "script": {"commands": commands},
        "checks": verify(spec, body),
        "metrics": {
            "size_mm": [round(abs(h - l), 2) for l, h in zip(low, high)],
            "volume_cm3": round(float(body["volume_mm3"]) / 1000, 3),
            "mass_g": round(float(body["mass_g"]), 1),
            "material": MATERIALS_RU.get(str(body.get("material")), str(body.get("material"))),
        },
        "images": {name: base64.b64encode(data).decode("ascii") for name, data in run.images.items()},
        "step_base64": base64.b64encode(run.step).decode("ascii"),
    }
