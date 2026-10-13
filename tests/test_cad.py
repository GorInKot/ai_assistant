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


# ------------------------------ команды CAD ------------------------------

ALLOWED_COMMANDS = {"solid.box", "solid.cylinder", "solid.fillet", "solid.hole", "material.assign"}


def test_commands_use_only_allowed_cad_commands():
    for part in (PLATE | dict(corner_radius=5), FLANGE):
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
def test_generate_maps_bad_spec_to_422(client, monkeypatch):
    monkeypatch.setattr(state.llm_service, "complete_text", ScriptedLLM("мусор", "мусор", "мусор").complete_text)
    headers = register(client, "cad4@test.local")
    resp = client.post("/api/cad/generate", headers=headers, json={"description": "Пластина"})
    assert resp.status_code == 422 and "Не удалось составить" in resp.json()["detail"]
