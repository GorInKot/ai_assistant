"""Чертёж трассы в API: построение, история по типам, пересборка без модели, «пока не поддерживается»."""

from __future__ import annotations

import base64
import dataclasses
import json

import pytest

import app.routers.cad as cad_router
import app.state as state
from app.db import SessionLocal
from app.services import cad_demand, route_drawing as rd
from tests.conftest import register
from tests.test_route_drawing import CADCRAFT_DIR, needs_cli

SPEC = {"title": "План трассы", "diameter_mm": 219, "wall_mm": 8,
        "segments": [{"length_m": 12, "turn_deg": 0}, {"length_m": 8, "turn_deg": 90},
                     {"length_m": 10, "turn_deg": -45}]}
DESCRIPTION = "Труба Ø219×8: 12 м, налево на 90° и 8 м, направо на 45° и 10 м"


@pytest.fixture
def llm(monkeypatch):
    replies: list[str] = []
    monkeypatch.setattr(state.llm_service, "complete_text", lambda messages, json_mode=False: replies.pop(0))
    return replies.append


def _route(client, headers, description=DESCRIPTION):
    return client.post("/api/cad/route", headers=headers, json={"description": description})


def _history(client, headers, kind):
    resp = client.get(f"/api/cad/history?kind={kind}", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["items"]


# ------------------------------ извлечение схемы из слов ------------------------------


class Scripted:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete_text(self, messages, *, json_mode=False):
        self.calls.append(messages)
        return self.replies.pop(0)


def test_extract_spec_reads_json_and_notes_assumed_material():
    result = rd.extract_spec("Труба 219x8, 12 м и 8 м под прямым углом", Scripted(json.dumps(SPEC)))
    assert result.spec.diameter_mm == 219 and len(result.spec.segments) == 3 and result.attempts == 1
    assert any("Материал" in a for a in result.assumptions)
    named = rd.extract_spec("Стальная труба 219x8", Scripted(json.dumps(SPEC)))
    assert named.assumptions == []


def test_extract_spec_retries_with_the_validation_error():
    crossing = {**SPEC, "segments": [{"length_m": 10, "turn_deg": 0}, {"length_m": 10, "turn_deg": 90},
                                       {"length_m": 5, "turn_deg": 90}, {"length_m": 15, "turn_deg": 90}]}
    llm = Scripted(json.dumps(crossing), json.dumps(SPEC))
    result = rd.extract_spec("Труба с петлёй", llm)
    assert result.attempts == 2 and "пересекает сама себя" in llm.calls[1][-1]["content"]


def test_extract_spec_passes_clarify_unsupported_and_skipped_through():
    assert rd.extract_spec("Нарисуй трассу", Scripted('{"clarify": "Укажите диаметр."}')).clarify == "Укажите диаметр."
    assert rd.extract_spec("Тройник", Scripted('{"unsupported": "тройники."}')).unsupported == "тройники"
    skipped = rd.extract_spec("Стальная труба с задвижкой", Scripted(json.dumps({**SPEC, "skipped": ["задвижка", "отметки 12,5 м"]})))
    assert "Пока не поддерживается и не учтено: задвижка." in skipped.assumptions  # размеры в «не учтено» отброшены


def test_extract_spec_gives_up_after_three_attempts():
    with pytest.raises(rd.RouteSpecError):
        rd.extract_spec("Труба", Scripted("мусор", "{}", '{"diameter_mm": 5}'))


# ------------------------------ API ------------------------------


def test_route_endpoints_require_login(client):
    assert client.post("/api/cad/route", json={"description": "труба"}).status_code == 401
    assert client.get("/api/cad/history?kind=route").status_code == 401


def test_route_reports_missing_cadcraft_and_unconfigured_model(client, monkeypatch, tmp_path):
    headers = register(client, "rt0@test.local")
    assert _route(client, headers).status_code == 503  # модель в тестах не настроена
    monkeypatch.setattr(cad_router, "settings", dataclasses.replace(state.settings, cadcraft_dir=tmp_path))
    resp = _route(client, headers)
    assert resp.status_code == 503 and "fetch_cadcraft" in resp.json()["detail"]
    assert client.get("/api/cad/status", headers=headers).json()["drawing"]["available"] is False
    assert _history(client, headers, "route") == []  # недоступность в историю не пишется


def test_history_kind_is_validated(client):
    headers = register(client, "rt1@test.local")
    assert client.get("/api/cad/history?kind=weird", headers=headers).status_code == 422


@needs_cli
def test_route_is_built_checked_and_returned_with_files(client, llm):
    headers = register(client, "rt2@test.local")
    llm(json.dumps(SPEC))
    resp = _route(client, headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["clarify"] is None and body["metrics"] == {"total_m": 30.0, "scale": "1:100", "segments": 3}
    assert all(c["ok"] for c in body["checks"]), body["checks"]
    assert base64.b64decode(body["pdf_base64"]).startswith(b"%PDF")
    assert b"AC1015" in base64.b64decode(body["dxf_base64"])[:400]  # DXF версии R2000
    assert base64.b64decode(body["dwg_base64"]).startswith(b"AC1015")  # DWG версии R2000
    assert base64.b64decode(body["images"]["preview"]).startswith(b"\x89PNG")
    assert body["summary"][0].startswith("Труба Ø219×8")


@needs_cli
def test_route_and_part_histories_are_separate(client, llm):
    headers = register(client, "rt3@test.local")
    llm(json.dumps(SPEC))
    _route(client, headers)
    assert [i["kind"] for i in _history(client, headers, "route")] == ["route"]
    assert _history(client, headers, "part") == []
    assert client.get("/api/cad/history", headers=headers).json()["items"] == []  # по умолчанию — детали


@needs_cli
def test_route_outcomes_are_recorded(client, llm):
    headers = register(client, "rt4@test.local")
    llm('{"clarify": "Укажите диаметр."}')
    _route(client, headers, "Нарисуй трассу")
    llm('{"unsupported": "тройники и ответвления"}')
    assert _route(client, headers, "Трасса с тройником").json()["unsupported"] == "тройники и ответвления"
    llm("мусор"), llm("мусор"), llm("мусор")
    assert _route(client, headers, "Труба").status_code == 422
    outcomes = [i["outcome"] for i in _history(client, headers, "route")]
    assert outcomes == ["error", "unsupported", "clarify"]


@needs_cli
def test_route_rebuild_does_not_use_the_model(client, llm, monkeypatch):
    headers = register(client, "rt5@test.local")
    llm(json.dumps(SPEC))
    first = _route(client, headers).json()
    item = _history(client, headers, "route")[0]
    assert item["can_rebuild"] is True

    def model_must_not_be_called(*args, **kwargs):
        raise AssertionError("пересборка не должна обращаться к модели")

    monkeypatch.setattr(state.llm_service, "complete_text", model_must_not_be_called)
    resp = client.post(f"/api/cad/history/{item['id']}/rebuild", headers=headers)
    assert resp.status_code == 200, resp.text
    again = resp.json()
    assert again["metrics"] == first["metrics"] and again["summary"] == first["summary"]
    assert all(c["ok"] for c in again["checks"])


@needs_cli
def test_unsupported_route_requests_reach_the_owner_demand_report(client, llm):
    headers = register(client, "rt6@test.local")
    llm('{"unsupported": "тройники и ответвления"}')
    _route(client, headers, "Трасса с тройником")
    with SessionLocal() as db:
        report = cad_demand.build(db, days=30)
    assert [t["topic"] for t in report["topics"]] == ["тройники и ответвления"]
