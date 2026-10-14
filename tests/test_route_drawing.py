"""План трассы трубопровода: геометрия, проверка схемы, команды CADCraft, самопроверка чертежа.

Тесты с настоящим cadcraft-cli пропускаются, если он не установлен (scripts/fetch_cadcraft.sh).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.services import route_drawing as rd

CADCRAFT_DIR = Path(__file__).resolve().parent.parent / "tools" / "cadcraft"
needs_cli = pytest.mark.skipif(not rd.is_available(CADCRAFT_DIR), reason="cadcraft-cli не установлен")

S = rd.Segment


def route(*segments, **kwargs) -> rd.RouteSpec:
    return rd.RouteSpec(diameter_mm=kwargs.pop("diameter_mm", 219), wall_mm=kwargs.pop("wall_mm", 8),
                        segments=list(segments), **kwargs)


SIMPLE = route(S(length_m=12), S(length_m=8, turn_deg=90), S(length_m=10, turn_deg=-45), S(length_m=14, turn_deg=-45))


# ------------------------------ геометрия ------------------------------


def test_geometry_follows_lengths_and_turns():
    geometry = rd.route_geometry(SIMPLE)
    assert geometry.total_m == 44
    assert geometry.points[1] == pytest.approx((12, 0))
    assert geometry.points[2] == pytest.approx((12, 8))  # поворот влево на 90°
    assert geometry.headings == pytest.approx([0, 90, 45, 0])
    assert geometry.points[3] == pytest.approx((12 + 10 * math.cos(math.radians(45)), 8 + 10 * math.sin(math.radians(45))))


def test_first_segment_turn_is_ignored():
    assert rd.route_geometry(route(S(length_m=5, turn_deg=90))).headings == [0.0]


@pytest.mark.parametrize("extent_m, expected", [(10, 50), (40, 200), (580, 2500), (610, 5000)])
def test_scale_is_the_smallest_standard_one_that_fits(extent_m, expected):
    scale = rd.choose_scale(rd.route_geometry(route(S(length_m=extent_m))))
    assert scale == expected


def test_huge_route_is_refused():
    with pytest.raises(rd.RouteSpecError):
        rd.choose_scale(rd.route_geometry(route(S(length_m=9000))))


# ------------------------------ проверка схемы ------------------------------


def test_valid_route_has_no_errors():
    assert rd.validate_spec(SIMPLE) == []


def test_empty_and_oversized_routes_are_refused():
    assert any("ни одного участка" in e for e in rd.validate_spec(route()))
    too_many = route(*[S(length_m=5, turn_deg=10) for _ in range(rd.MAX_SEGMENTS + 1)])
    assert any("слишком много" in e for e in rd.validate_spec(too_many))


def test_lengths_turns_and_diameters_are_range_checked():
    assert rd.validate_spec(route(S(length_m=0.1)))
    assert rd.validate_spec(route(S(length_m=5), S(length_m=5, turn_deg=0))) == []  # продолжение по прямой
    assert rd.validate_spec(route(S(length_m=5), S(length_m=5, turn_deg=180)))
    assert rd.validate_spec(route(S(length_m=5), diameter_mm=5))
    assert rd.validate_spec(route(S(length_m=5), wall_mm=200))


@pytest.mark.parametrize("third, fourth", [(5, 15), (10, 15)], ids=["crosses", "touches-start"])
def test_route_crossing_or_touching_itself_is_refused(third, fourth):
    # петля: четвёртый участок идёт вниз и пересекает первый (5 м) либо проходит через его начало (10 м)
    spec = route(S(length_m=10), S(length_m=10, turn_deg=90), S(length_m=third, turn_deg=90),
                 S(length_m=fourth, turn_deg=90))
    assert any("пересекает сама себя" in e for e in rd.validate_spec(spec))


def test_adjacent_segments_are_not_reported_as_crossing():
    assert rd.validate_spec(route(S(length_m=10), S(length_m=10, turn_deg=90), S(length_m=10, turn_deg=-90))) == []


# ------------------------------ команды ------------------------------


def test_commands_contain_one_dimension_per_segment_and_per_bend():
    commands, info = rd.build_commands(SIMPLE)
    assert info["dimensions_expected"] == 4 + 3
    assert sum(c.startswith("dimaligned ") for c in commands) == 4
    assert sum(c.startswith("dimangular ") for c in commands) == 3
    assert info["scale"] == 200 and info["total_m"] == 44


def test_dimension_texts_come_from_the_spec_not_from_the_sheet():
    commands, _ = rd.build_commands(SIMPLE)
    dims = [c for c in commands if c.startswith("dimaligned ")]
    assert [('"text": "%s"' % t) in d for t, d in zip(["12,00", "8,00", "10,00", "14,00"], dims)] == [True] * 4


def test_straight_continuation_gets_no_angle_dimension_and_is_shown_as_180():
    spec = route(S(length_m=12), S(length_m=8, turn_deg=90), S(length_m=14))  # «и ещё 14 м» по прямой
    commands, info = rd.build_commands(spec)
    assert sum(c.startswith("dimangular ") for c in commands) == 1
    assert info["dimensions_expected"] == 3 + 1
    assert any('"180°"' in c for c in commands)


def test_commands_use_only_known_drafting_commands():
    commands, _ = rd.build_commands(SIMPLE)
    allowed = {"layer.new", "layer.current", "style", "dimstyle", "rectang", "line", "text", "pline", "circle",
               "dimaligned", "dimangular"}
    assert {c.split(" ", 1)[0] for c in commands} <= allowed


def test_user_text_cannot_inject_extra_commands():
    hostile = route(S(length_m=5), title='x"}\nsaveas {"path": "/tmp/evil.dxf"}')
    commands, _ = rd.build_commands(hostile)
    # название попадает только внутрь JSON-строки одной команды text; переводов строк в команде нет
    assert all("\n" not in c for c in commands)
    assert all(c.split(" ", 1)[0] != "saveas" for c in commands)


# ------------------------------ самопроверка и настоящий CADCraft ------------------------------


def test_unicode_escapes_of_old_dxf_versions_are_decoded():
    # в DXF R2000 не-ASCII символы записаны как \\U+0422; «1–2» не должно «съесть» следующую цифру
    assert rd._dxf_text("\\U+0422\\U+0420\\U+0423\\U+0411\\U+0410") == "ТРУБА"
    assert rd._dxf_text("1\\U+20132") == "1–2"
    assert rd._dxf_text("PIPELINE") == "PIPELINE"


def test_layers_have_ascii_names():
    commands, _ = rd.build_commands(SIMPLE)
    layer_names = [c for c in commands if c.startswith("layer.new ")]
    assert len(layer_names) == 4
    assert all(c.split('"name": "')[1].split('"')[0].isascii() for c in layer_names)


def test_route_polyline_is_read_back_from_dxf_text():
    dxf = "\n".join(["  0", "LWPOLYLINE", "  8", "PIPELINE", " 90", "2", " 10", "1.5", " 20", "2.5", " 10", "3.0", " 20", "4.0",
                     "  0", "LWPOLYLINE", "  8", "FRAME", " 10", "0.0", " 20", "0.0", "  0", "EOF"])
    assert rd.read_route_polyline(dxf) == [(1.5, 2.5), (3.0, 4.0)]


@needs_cli
def test_real_cadcraft_builds_a_drawing_that_passes_all_checks():
    files = rd.draw_route(SIMPLE, CADCRAFT_DIR)
    assert files.pdf.startswith(b"%PDF") and files.png.startswith(b"\x89PNG") and b"AC1015" in files.dxf[:400]
    assert files.dwg.startswith(b"AC1015")  # DWG R2000: самая совместимая версия
    assert all(c["ok"] for c in files.checks), files.checks
    assert "DWG читается обратно" in [c["name"] for c in files.checks]
    assert files.info["scale"] == 200


@needs_cli
def test_check_catches_a_drawing_that_does_not_match_the_order():
    files = rd.draw_route(SIMPLE, CADCRAFT_DIR)
    longer = route(S(length_m=13), *SIMPLE.segments[1:])  # «заказали» 13 м первого участка, нарисовано 12
    checks = rd.verify_drawing(longer, files.info, {"counts": {"Dimension": 7, "Circle": 5}, "extents": [[0, 0], [420, 297]]},
                               files.dxf.decode("utf-8", errors="replace"))
    assert not next(c for c in checks if c["name"].startswith("Длины")).get("ok")


@needs_cli
def test_invalid_spec_never_reaches_cadcraft():
    with pytest.raises(rd.RouteSpecError):
        rd.draw_route(route(), CADCRAFT_DIR)


def test_missing_cadcraft_is_reported(tmp_path):
    with pytest.raises(rd.CadcraftUnavailable):
        rd.draw_route(SIMPLE, tmp_path)
