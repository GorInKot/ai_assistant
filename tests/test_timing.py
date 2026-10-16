"""Страница «Замер»: секундомер «вручную / с ассистентом», машинное время, сводка, CSV."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.services import timing
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, auth_headers, register
from tests.test_text_compare import LETTER, _docx

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
BASE = "/api/admin/timing"


@pytest.fixture
def admin(client):
    return auth_headers(client, ADMIN_EMAIL, ADMIN_PASSWORD)


class _Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def tick(self, seconds: float):
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(timing, "_now", c)
    return c


def _start(client, headers, **overrides):
    body = {"participant": "У1", "task": "spec_to_excel", "document": "Лист 3", "doc_type": "clean",
            "mode": "manual", **overrides}
    return client.post(f"{BASE}/runs", json=body, headers=headers)


def test_only_managers_and_admins(client):
    headers = register(client, "worker@test.local")
    assert client.get(BASE, headers=headers).status_code == 403
    assert _start(client, headers).status_code == 403


def test_one_running_run_at_a_time_and_results(client, admin, clock):
    run = _start(client, admin).json()
    assert run["status"] == "running"
    assert _start(client, admin).status_code == 400  # второй — нельзя

    clock.tick(300)
    stopped = client.post(f"{BASE}/runs/{run['id']}/stop", json={}, headers=admin).json()
    assert stopped["status"] == "done" and stopped["total_seconds"] == 300.0

    updated = client.patch(f"{BASE}/runs/{run['id']}", json={"sheets": 2, "positions": 18, "errors_wrong": 1,
                                                            "comment": "почерк неразборчив"}, headers=admin).json()
    assert (updated["sheets"], updated["errors_wrong"], updated["comment"]) == (2, 1, "почерк неразборчив")
    assert client.patch(f"{BASE}/runs/{run['id']}", json={"positions": -1}, headers=admin).status_code == 400
    assert client.post(f"{BASE}/runs/{run['id']}/stop", json={}, headers=admin).status_code == 400


def test_machine_time_accrues_only_in_assisted_mode(client, admin):
    files = [("files", ("a.docx", _docx(LETTER), DOCX)), ("files", ("b.docx", _docx(LETTER), DOCX))]

    manual = _start(client, admin, task="compare_versions", doc_type="text").json()
    client.post("/api/ask-files", data={"message": "сравни, старая — первая"}, files=files, headers=admin)
    manual = client.post(f"{BASE}/runs/{manual['id']}/stop", json={}, headers=admin).json()
    assert manual["machine_seconds"] == 0.0

    assisted = _start(client, admin, task="compare_versions", doc_type="text", mode="assisted").json()
    client.post("/api/ask-files", data={"message": "сравни, старая — первая"}, files=files, headers=admin)
    active = client.get(f"{BASE}/active", headers=admin).json()["active"]
    assert active["id"] == assisted["id"] and active["human_seconds"] <= active["total_seconds"]
    from app.db import SessionLocal, TimingRun

    with SessionLocal() as db:  # в ответе API — до десятых, короткое сравнение даёт сотые доли
        assert db.get(TimingRun, assisted["id"]).machine_seconds > 0


def test_summary_medians_per_sheet_and_saving(client, admin, clock):
    for mode, seconds, sheets in (("manual", 600, 2), ("manual", 900, 3), ("manual", 400, 1),
                                  ("assisted", 120, 2), ("assisted", 150, 3), ("assisted", 90, 1)):
        run = _start(client, admin, mode=mode).json()
        clock.tick(seconds)
        client.post(f"{BASE}/runs/{run['id']}/stop", json={}, headers=admin)
        client.patch(f"{BASE}/runs/{run['id']}", json={"sheets": sheets}, headers=admin)
    failed = _start(client, admin, mode="manual").json()
    clock.tick(5000)
    client.post(f"{BASE}/runs/{failed['id']}/stop", json={"failed": True}, headers=admin)  # сбой — не в счёт

    row = next(r for r in client.get(BASE, headers=admin).json()["summary"] if r["task"] == "spec_to_excel")
    assert row["unit"] == "лист"
    assert row["manual"] == {"n": 3, "median": 300.0, "min": 300.0, "max": 400.0}  # 300, 300, 400 с на лист
    assert row["assisted"]["median"] == 60.0  # 60, 50, 90
    assert row["saved_seconds"] == 240.0 and row["saved_share"] == 0.8


def test_csv_opens_in_russian_excel_and_does_not_run_formulas(client, admin, clock):
    run = _start(client, admin, document="=HYPERLINK(\"http://x\")").json()
    clock.tick(75.5)
    client.post(f"{BASE}/runs/{run['id']}/stop", json={}, headers=admin)
    resp = client.get(f"{BASE}/export", headers=admin)
    assert resp.status_code == 200 and resp.content.startswith("﻿".encode("utf-8"))
    lines = resp.content.decode("utf-8-sig").splitlines()
    assert lines[0].startswith("№;участник;задача;документ")
    assert "Перенос спецификации в Excel" in lines[1] and ";вручную;" in lines[1]
    assert "'=HYPERLINK" in lines[1] and ";75,5;" in lines[1]


def test_runs_are_private_to_their_operator(client, admin):
    run = _start(client, admin).json()
    other = register(client, "manager2@test.local")
    from app.db import ROLE_MANAGER, Role, SessionLocal, User

    with SessionLocal() as db:
        user = db.query(User).filter(User.email == "manager2@test.local").one()
        user.roles = [db.query(Role).filter(Role.name == ROLE_MANAGER).one()]
        db.commit()
    assert client.get(BASE, headers=other).json()["runs"] == []
    assert client.post(f"{BASE}/runs/{run['id']}/stop", json={}, headers=other).status_code == 404
