"""Конструктор деталей: проверка спецификации, команды CAD, самопроверка, API.

Тесты, которым нужен настоящий solidcraft-cli, пропускаются, если он не установлен
(scripts/fetch_solidcraft.sh); остальные работают без него и без модели.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math

import pytest

import app.routers.cad as cad_router
import app.state as state
from app.services import cad
from tests.conftest import register

needs_cli = pytest.mark.skipif(
    not cad.is_available(state.settings.solidcraft_dir), reason="solidcraft-cli не установлен"
)

PLATE = dict(shape="box", length=100, width=60, height=5, material="Steel",
             hole_grid=dict(count_x=2, count_y=2, offset_x=10, offset_y=10, diameter=6))
PIPES = dict(shape="pipes", diameter=100, wall_thickness=5, length=500, length2=500, angle_deg=60,
             material="Steel")
FLANGE = dict(shape="cylinder", diameter=150, height=12, bore_diameter=60, material="Stainless Steel",
              bolt_circle=dict(count=6, circle_diameter=110, diameter=14))


def spec(**kwargs) -> cad.PartSpec:
    return cad.PartSpec.model_validate(kwargs)


# ------------------------------ проверка спецификации ------------------------------


def test_valid_specs_pass():
    assert cad.validate_spec(spec(**PLATE)) == []
    assert cad.validate_spec(spec(**FLANGE)) == []


def test_missing_and_out_of_range_dimensions_are_reported():
    errors = cad.validate_spec(spec(shape="box", length=100, height=5))
    assert any("ширина" in e for e in errors)
    errors = cad.validate_spec(spec(shape="box", length=100, width=60, height=50000))
    assert any("вне допустимого диапазона" in e for e in errors)


def test_fields_of_the_other_shape_are_rejected():
    assert cad.validate_spec(spec(shape="box", length=10, width=10, height=2, diameter=5))
    assert cad.validate_spec(spec(shape="cylinder", diameter=10, height=2, length=5))


def test_hole_outside_the_part_is_reported():
    errors = cad.validate_spec(spec(shape="box", length=100, width=60, height=5,
                                    holes=[dict(x=98, y=30, diameter=6)]))
    assert len(errors) == 1 and "выходит за границы" in errors[0]


def test_overlapping_holes_and_bore_conflicts_are_reported():
    errors = cad.validate_spec(spec(shape="box", length=100, width=60, height=5,
                                    holes=[dict(x=30, y=30, diameter=10), dict(x=34, y=30, diameter=10)]))
    assert any("пересекаются" in e for e in errors)
    errors = cad.validate_spec(spec(shape="cylinder", diameter=100, height=5, bore_diameter=60,
                                    bolt_circle=dict(count=4, circle_diameter=64, diameter=10)))
    assert any("центральным отверстием" in e for e in errors)
    errors = cad.validate_spec(spec(shape="cylinder", diameter=100, height=5, bore_diameter=120))
    assert any("центрального отверстия" in e for e in errors)


def test_corner_radius_must_fit():
    errors = cad.validate_spec(spec(shape="box", length=100, width=20, height=5, corner_radius=10))
    assert errors and "радиус" in errors[0]


def test_too_many_holes_are_refused():
    errors = cad.validate_spec(spec(shape="box", length=1000, width=1000, height=5,
                                    hole_grid=dict(count_x=20, count_y=20, offset_x=10, offset_y=10, diameter=4)))
    assert errors and "слишком много" in errors[0]


def test_hole_grid_and_bolt_circle_positions():
    corners = cad.hole_positions(spec(**PLATE))
    assert sorted((round(x), round(y)) for x, y, _ in corners) == [(10, 10), (10, 50), (90, 10), (90, 50)]
    flange = cad.hole_positions(spec(**FLANGE))
    assert len(flange) == 6 and all(math.isclose(math.hypot(x, y), 55) for x, y, _ in flange)
    single = cad.hole_positions(spec(shape="box", length=100, width=60, height=5,
                                     hole_grid=dict(count_x=1, count_y=1, diameter=10)))
    assert single == [(50.0, 30.0, 10.0)]


# ------------------------------ трубы ------------------------------


def test_pipe_joint_spec_is_valid():
    assert cad.validate_spec(spec(**PIPES)) == []
    assert cad.validate_spec(spec(**(PIPES | dict(length2=None)))) == []  # вторая труба равна первой


def test_pipe_joint_requires_angle_wall_and_diameter():
    errors = cad.validate_spec(spec(shape="pipes", diameter=100, length=500))
    assert any("толщина стенки" in e for e in errors) and any("угол" in e for e in errors)


@pytest.mark.parametrize("angle", [5, 175])
def test_pipe_angle_out_of_range_is_refused(angle):
    errors = cad.validate_spec(spec(**(PIPES | dict(angle_deg=angle))))
    assert any("угол между трубами должен быть" in e for e in errors)


def test_pipe_wall_must_be_thinner_than_radius_and_other_fields_are_refused():
    assert any("меньше радиуса" in e for e in cad.validate_spec(spec(**(PIPES | dict(wall_thickness=50)))))
    errors = cad.validate_spec(spec(**(PIPES | dict(holes=[dict(x=1, y=1, diameter=5)]))))
    assert any("shape=pipes" in e for e in errors)


def test_short_pipe_leg_names_the_minimum_length():
    errors = cad.validate_spec(spec(**(PIPES | dict(length=50))))
    assert len(errors) == 1 and "слишком короткая" in errors[0] and "137" in errors[0]


def test_box_and_cylinder_refuse_pipe_only_fields():
    assert cad.validate_spec(spec(**(PLATE | dict(angle_deg=60))))
    assert cad.validate_spec(spec(shape="cylinder", diameter=100, height=5, length2=10))


def test_cylinder_wall_thickness_defines_the_bore():
    pipe = spec(shape="cylinder", diameter=60, wall_thickness=4, height=1000)
    assert cad.validate_spec(pipe) == [] and cad.effective_bore(pipe) == 52
    bore_command = cad.spec_to_commands(pipe)[1]["params"]
    assert bore_command["diameter"] == 52 and bore_command["operation"] == "cut"
    assert cad.validate_spec(spec(shape="cylinder", diameter=60, wall_thickness=4, bore_diameter=40, height=10))
    assert "Ø52" in " ".join(cad.describe(pipe))


def test_pipe_joint_geometry_matches_closed_forms():
    right_angle = cad.expected_geometry(spec(**(PIPES | dict(angle_deg=90))))
    assert [round(v, 3) for v in right_angle["size"]] == [100.0, 550.0, 550.0]
    assert math.isclose(right_angle["volume"], math.pi * (50**2 - 45**2) * 1000)
    assert "Угол между трубами 60°" in " ".join(cad.describe(spec(**PIPES)))


def test_pipe_path_points_follow_angle_and_leg_lengths():
    points = cad._pipe_path_points(spec(**(PIPES | dict(angle_deg=90, length=800, length2=300))))
    assert points == [[800.0, 0], [0, 0], [0.0, 300.0]]


# ------------------------------ команды CAD ------------------------------

ALLOWED_COMMANDS = {"solid.box", "solid.cylinder", "solid.fillet", "solid.hole", "material.assign",
                    "sketch.create", "sketch.line", "sketch.finish", "solid.pipe"}


def test_commands_use_only_allowed_cad_commands():
    for part in (PLATE | dict(corner_radius=5), FLANGE, PIPES):
        commands = cad.spec_to_commands(spec(**part))
        assert {c["command"] for c in commands} <= ALLOWED_COMMANDS
        assert commands[-1]["command"] == "material.assign"


def test_model_text_never_reaches_cad_commands():
    hostile = "x\"; file.export /etc/passwd ../../"
    commands = cad.spec_to_commands(spec(**(PLATE | dict(title=hostile, material=hostile))))
    dumped = json.dumps(commands)
    assert "passwd" not in dumped and "file." not in dumped
    # неизвестный материал заменяется материалом по умолчанию
    assert commands[-1]["params"]["material"] == "Steel"


def test_material_names_are_normalized():
    assert cad.normalize_material("Aluminum") == ("Aluminum", False)
    assert cad.normalize_material("алюминий") == ("Aluminum", False)
    assert cad.normalize_material("что-то странное") == ("Steel", True)


def test_expected_geometry_matches_hand_calculation():
    plate = cad.expected_geometry(spec(**(PLATE | dict(corner_radius=5))))
    expected = 100 * 60 * 5 - 4 * (1 - math.pi / 4) * 25 * 5 - 4 * math.pi * 9 * 5
    assert math.isclose(plate["volume"], expected) and plate["size"] == [5, 60, 100]


def test_description_lists_what_was_understood():
    lines = cad.describe(spec(**FLANGE))
    assert lines[0].startswith("Цилиндр Ø150") and "6 × Ø14" in " ".join(lines)
    assert lines[-1] == "Материал: нержавеющая сталь"


# ------------------------------ модель → спецификация ------------------------------


class ScriptedLLM:
    """Отвечает заготовленными строками по очереди и запоминает, что у него спрашивали."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def complete_text(self, messages, *, json_mode=False):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


def test_extract_spec_accepts_json_in_code_fence():
    llm = ScriptedLLM("```json\n" + json.dumps(PLATE) + "\n```")
    result = cad.extract_spec("Стальная пластина 100 на 60", llm)
    assert result.spec.length == 100 and result.attempts == 1 and result.assumptions == []


def test_extract_spec_retries_with_error_text_and_then_succeeds():
    broken = json.dumps(PLATE | dict(holes=[dict(x=99, y=30, diameter=6)]))
    llm = ScriptedLLM(broken, json.dumps(PLATE))
    result = cad.extract_spec("Стальная пластина 100 на 60", llm)
    assert result.attempts == 2
    feedback = llm.calls[1][-1]["content"]
    assert "выходит за границы" in feedback and "Исправь" in feedback


def test_extract_spec_gives_up_after_three_attempts():
    llm = ScriptedLLM("не json", "{}", json.dumps(dict(shape="box")))
    with pytest.raises(cad.CadSpecError):
        cad.extract_spec("Пластина", llm)
    assert len(llm.calls) == 3


def test_extract_spec_passes_clarifying_question_through():
    llm = ScriptedLLM(json.dumps({"clarify": "Укажите диаметр втулки."}))
    result = cad.extract_spec("Сделай втулку", llm)
    assert result.clarify == "Укажите диаметр втулки." and result.spec is None


def test_extract_spec_notes_assumed_material():
    llm = ScriptedLLM(json.dumps(PLATE))
    assert cad.extract_spec("Пластина 100 на 60, толщина 5", llm).assumptions
    llm = ScriptedLLM(json.dumps(PLATE))
    assert cad.extract_spec("Стальная пластина 100 на 60, толщина 5", llm).assumptions == []


def test_extract_spec_reports_fields_that_are_not_in_the_schema():
    ignored = json.dumps(PLATE | dict(chamfer=2, description="пластина"))
    result = cad.extract_spec("Стальная пластина 100 на 60", ScriptedLLM(ignored))
    assert any("chamfer" in a for a in result.assumptions)
    assert not any("description" in a for a in result.assumptions)  # безобидное поле не упоминаем


def test_extract_spec_passes_unsupported_request_through():
    llm = ScriptedLLM(json.dumps({"unsupported": "тройники и ответвления труб."}))
    result = cad.extract_spec("Сделай тройник", llm)
    assert result.unsupported == "тройники и ответвления труб" and result.spec is None and result.clarify is None


def test_extract_spec_lists_skipped_features_as_not_supported_yet():
    llm = ScriptedLLM(json.dumps(PLATE | dict(skipped=["фаски по кромкам", "резьба"])))
    result = cad.extract_spec("Стальная пластина с фасками", llm)
    assert result.spec is not None
    assert "Пока не поддерживается и не учтено: фаски по кромкам, резьба." in result.assumptions


def test_extract_spec_accepts_pipe_joint_json():
    result = cad.extract_spec("Две стальные трубы под углом 60", ScriptedLLM(json.dumps(PIPES)))
    assert result.spec.shape == "pipes" and result.spec.angle_deg == 60


def test_extract_spec_rejects_empty_and_huge_descriptions():
    with pytest.raises(cad.CadSpecError):
        cad.extract_spec("   ", ScriptedLLM())
    with pytest.raises(cad.CadSpecError):
        cad.extract_spec("а" * (cad.MAX_DESCRIPTION + 1), ScriptedLLM())


# ------------------------------ настоящий SolidCraft ------------------------------


@needs_cli
@pytest.mark.parametrize("part", [PLATE | dict(corner_radius=5), FLANGE], ids=["plate", "flange"])
def test_real_cad_builds_part_and_matches_calculation(part):
    s = spec(**part)
    run = cad.run_cad(cad.spec_to_commands(s), state.settings.solidcraft_dir)
    assert run.step.startswith(b"ISO-10303") and run.images["iso"].startswith(b"\x89PNG")
    assert all(c["ok"] for c in cad.verify(s, run.body)), cad.verify(s, run.body)
    assert run.body["mass_g"] > 0


@needs_cli
@pytest.mark.parametrize("angle", [60, 90, 120])
def test_real_cad_builds_pipe_joint_and_matches_formulas(angle):
    s = spec(**(PIPES | dict(angle_deg=angle, length=400, length2=300)))
    run = cad.run_cad(cad.spec_to_commands(s), state.settings.solidcraft_dir)
    assert all(c["ok"] for c in cad.verify(s, run.body)), cad.verify(s, run.body)
    assert run.body["bbox"]["max"][2] - run.body["bbox"]["min"][2] == pytest.approx(100, abs=0.01)


@needs_cli
def test_verification_catches_a_wrong_model():
    built = spec(**PLATE)
    run = cad.run_cad(cad.spec_to_commands(built), state.settings.solidcraft_dir)
    claimed = spec(**(PLATE | dict(height=6)))  # «заказали» 6 мм, построено 5
    checks = cad.verify(claimed, run.body)
    assert not any(c["ok"] for c in checks)


@needs_cli
def test_cad_failure_is_reported_as_cad_error():
    with pytest.raises(cad.CadError):
        cad.run_cad([{"command": "solid.no_such_command", "params": {}}], state.settings.solidcraft_dir)


# ------------------------------ API ------------------------------


def test_cad_endpoints_require_login(client):
    assert client.get("/api/cad/status").status_code == 401
    assert client.post("/api/cad/generate", json={"description": "пластина"}).status_code == 401


def test_status_reports_missing_solidcraft(client, monkeypatch, tmp_path):
    monkeypatch.setattr(cad_router, "settings", dataclasses.replace(state.settings, solidcraft_dir=tmp_path))
    headers = register(client, "cad0@test.local")
    body = client.get("/api/cad/status", headers=headers).json()
    assert body == {"available": False, "version": None, "editor": False}
    resp = client.post("/api/cad/generate", headers=headers, json={"description": "пластина"})
    assert resp.status_code == 503 and "fetch_solidcraft" in resp.json()["detail"]


def test_generate_reports_unconfigured_model(client):
    headers = register(client, "cad1@test.local")
    resp = client.post("/api/cad/generate", headers=headers, json={"description": "Пластина 100 на 60, 5 мм"})
    assert resp.status_code == 503  # в тестах LLM не настроен


@needs_cli
def test_generate_returns_model_checks_and_files(client, monkeypatch):
    monkeypatch.setattr(state.llm_service, "complete_text", ScriptedLLM(json.dumps(PLATE)).complete_text)
    headers = register(client, "cad2@test.local")
    resp = client.post("/api/cad/generate", headers=headers,
                       json={"description": "Стальная пластина 100 на 60 мм, толщина 5, четыре отверстия Ø6"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["clarify"] is None and body["attempts"] == 1
    assert all(c["ok"] for c in body["checks"])
    assert body["metrics"]["size_mm"] == [100.0, 60.0, 5.0] and body["metrics"]["material"] == "сталь"
    assert base64.b64decode(body["step_base64"]).startswith(b"ISO-10303")
    assert base64.b64decode(body["images"]["iso"]).startswith(b"\x89PNG")
    assert body["script"]["commands"][0]["command"] == "solid.box"


@needs_cli
def test_generate_returns_clarifying_question(client, monkeypatch):
    monkeypatch.setattr(state.llm_service, "complete_text",
                        ScriptedLLM(json.dumps({"clarify": "Укажите толщину."})).complete_text)
    headers = register(client, "cad3@test.local")
    body = client.post("/api/cad/generate", headers=headers, json={"description": "Пластина 100 на 60"}).json()
    assert body == {"clarify": "Укажите толщину.", "attempts": 1}


@needs_cli
def test_generate_says_not_supported_yet_and_lists_what_can_be_built(client, monkeypatch):
    monkeypatch.setattr(state.llm_service, "complete_text",
                        ScriptedLLM(json.dumps({"unsupported": "тройники и ответвления труб"})).complete_text)
    headers = register(client, "cad5@test.local")
    resp = client.post("/api/cad/generate", headers=headers, json={"description": "Сделай тройник"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["unsupported"] == "тройники и ответвления труб" and body["clarify"] is None
    assert body["can_build"] == cad.CAPABILITIES and "images" not in body


@needs_cli
def test_generate_maps_bad_spec_to_422(client, monkeypatch):
    monkeypatch.setattr(state.llm_service, "complete_text", ScriptedLLM("мусор", "мусор", "мусор").complete_text)
    headers = register(client, "cad4@test.local")
    resp = client.post("/api/cad/generate", headers=headers, json={"description": "Пластина"})
    assert resp.status_code == 422 and "Не удалось составить" in resp.json()["detail"]
