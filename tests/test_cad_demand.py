"""Спрос на конструктор деталей: что просили построить, но это пока не поддерживается."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from io import BytesIO

from openpyxl import load_workbook

from app.db import CadRequest, SessionLocal, User
from app.services import cad, cad_demand
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, auth_headers, register

SKIPPED = cad.SKIPPED_NOTE_PREFIX


def _user_id(email: str) -> int:
    with SessionLocal() as db:
        return db.query(User).filter(User.email == email).one().id


def _add(db, user_id, description, outcome, *, detail=None, notes=None, created_at=None):
    row = CadRequest(
        user_id=user_id, description=description, outcome=outcome, detail=detail,
        spec_json='{"shape": "box"}' if outcome == "built" else None,
        notes_json=json.dumps(notes, ensure_ascii=False) if notes else None,
    )
    if created_at:
        row.created_at = created_at
    db.add(row)


def test_refusals_are_grouped_and_counted_by_distinct_people(client):
    register(client, "d1@test.local", division="ЦА")
    register(client, "d2@test.local", division="Филиал")
    first, second = _user_id("d1@test.local"), _user_id("d2@test.local")
    with SessionLocal() as db:
        _add(db, first, "Сделай тройник", "unsupported", detail="тройники и ответвления труб")
        _add(db, first, "Тройник из труб 150", "unsupported", detail="тройники и ответвления")
        _add(db, second, "Тройник 100 мм", "unsupported", detail="тройники и ответвления труб")
        _add(db, second, "Шестерня на 20 зубьев", "unsupported", detail="шестерни")
        _add(db, first, "Пластина 100 на 60", "built")
        db.commit()
        report = cad_demand.build(db, days=30)

    assert report["requests"] == 5 and report["unsupported"] == 4 and report["built"] == 1
    top, other = report["topics"]
    assert top["topic"].startswith("тройники") and top["people"] == 2 and top["requests"] == 3
    assert top["divisions"] == ["Филиал", "ЦА"] and top["kind"] == "refused"
    assert other["topic"] == "шестерни" and other["people"] == 1


def test_skipped_features_of_built_parts_count_and_merge_with_refusals(client):
    register(client, "d3@test.local")
    register(client, "d4@test.local")
    first, second = _user_id("d3@test.local"), _user_id("d4@test.local")
    with SessionLocal() as db:
        _add(db, first, "Пластина с фасками", "built", notes=[SKIPPED + "фаски по кромкам, резьба."])
        _add(db, second, "Болт М10 с резьбой", "unsupported", detail="резьба")
        db.commit()
        report = cad_demand.build(db, days=30)

    topics = {t["topic"]: t for t in report["topics"]}
    assert set(topics) == {"резьба", "фаски по кромкам"}
    assert topics["резьба"]["people"] == 2 and topics["резьба"]["kind"] == "refused"
    assert topics["резьба"]["refused"] == 1 and topics["резьба"]["partial"] == 1
    assert topics["фаски по кромкам"]["kind"] == "partial"
    assert topics["фаски по кромкам"]["examples"] == ["Пластина с фасками"]


def test_the_same_request_with_two_skipped_features_is_counted_once_per_topic(client):
    register(client, "d5@test.local")
    with SessionLocal() as db:
        _add(db, _user_id("d5@test.local"), "Пластина с фасками и пазами", "built",
             notes=[SKIPPED + "фаски, пазы."])
        db.commit()
        report = cad_demand.build(db, days=30)
    assert sorted(t["topic"] for t in report["topics"]) == ["пазы", "фаски"]
    assert all(t["requests"] == 1 and t["people"] == 1 for t in report["topics"])


def test_old_requests_and_requests_without_gaps_are_ignored(client):
    register(client, "d6@test.local")
    user = _user_id("d6@test.local")
    with SessionLocal() as db:
        _add(db, user, "Старый тройник", "unsupported", detail="тройники",
             created_at=datetime.now(timezone.utc) - timedelta(days=100))
        _add(db, user, "Вопрос без размеров", "clarify", detail="Укажите длину")
        _add(db, user, "Пластина", "built")
        db.commit()
        report = cad_demand.build(db, days=30)
    assert report["topics"] == [] and report["requests"] == 2 and report["clarify"] == 1


def test_empty_report_is_well_formed(client):
    with SessionLocal() as db:
        report = cad_demand.build(db, days=30)
    assert report["requests"] == 0 and report["topics"] == [] and report["grouping"] == "words"


def test_report_exposes_no_identity(client):
    register(client, "private.person@test.local")
    with SessionLocal() as db:
        _add(db, _user_id("private.person@test.local"), "Сделай тройник", "unsupported", detail="тройники")
        db.commit()
        dumped = json.dumps(cad_demand.build(db, days=30), ensure_ascii=False, default=str)
    assert "private.person" not in dumped and "user_id" not in dumped


def test_gaps_endpoint_includes_constructor_section_for_admin_only(client):
    user = register(client, "d7@test.local")
    with SessionLocal() as db:
        _add(db, _user_id("d7@test.local"), "Сделай тройник", "unsupported", detail="тройники")
        db.commit()
    assert client.get("/api/admin/knowledge-gaps", headers=user).status_code == 403
    admin = auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    body = client.get("/api/admin/knowledge-gaps", headers=admin).json()
    assert body["cad"]["topics"][0]["topic"] == "тройники"


def test_excel_export_has_constructor_sheet_and_neutralizes_formulas(client):
    register(client, "d8@test.local")
    with SessionLocal() as db:
        _add(db, _user_id("d8@test.local"), '=HYPERLINK("http://evil.example","клик")', "unsupported",
             detail='=1+1 тройники')
        db.commit()
    admin = auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    resp = client.get("/api/admin/knowledge-gaps/export", headers=admin)
    sheet = load_workbook(BytesIO(resp.content))["Конструктор"]
    assert sheet["A1"].value == "Что просили, но пока не поддерживается"
    assert sheet["A2"].data_type == "s" and sheet["A2"].quotePrefix is True
    assert sheet["F2"].data_type == "s"  # пример запроса с «=» не становится формулой


def test_note_with_parenthetical_comment_is_not_split_into_fragments(client):
    # реальная запись: модель положила в «не учтено» размеры с пояснением в скобках
    register(client, "d9@test.local")
    with SessionLocal() as db:
        _add(db, _user_id("d9@test.local"), "Две трубы под углом 60°. Габариты 800 × 534 × 100 мм", "built",
             notes=[SKIPPED + "габариты 800 × 534 × 100 мм (не учитывается в конструкции, только геометрия стыка "
                    "под углом 60°)."])
        _add(db, _user_id("d9@test.local"), "Пластина с фасками и резьбой", "built",
             notes=[SKIPPED + "фаски (по кромкам, 2 мм), резьба."])
        db.commit()
        report = cad_demand.build(db, days=30)
    assert sorted(t["topic"] for t in report["topics"]) == ["резьба", "фаски"]
