"""Закрытый контур: ни один путь сотрудника не выходит за пределы машины; модели вне контура
отключаются при старте."""

from __future__ import annotations

import socket

import pytest

from app import netguard
from tests.conftest import register
from tests.test_spec_registry import _synthetic_scan, needs_ocr
from tests.test_text_compare import LETTER, _docx

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_guard_blocks_outside_and_lets_loopback_through():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        with netguard.no_external_connections() as log:
            socket.create_connection(server.getsockname(), timeout=2).close()
            with pytest.raises(OSError):
                socket.create_connection(("8.8.8.8", 53), timeout=1)
            with pytest.raises(OSError):
                socket.create_connection(("example.com", 80), timeout=1)  # даже имя не разрешается
    finally:
        server.close()
    assert len(log.local) == 1 and log.blocked == ["8.8.8.8:53", "DNS example.com"]
    socket.getaddrinfo("localhost", 80)  # после блока всё восстановлено


@pytest.mark.parametrize("base_url, api_key, allowed", [
    ("http://127.0.0.1:11434/v1", None, True),
    ("http://localhost:11434/v1", None, True),
    ("http://10.20.30.40:8000/v1", None, True),         # GPU-сервер во внутренней сети
    ("https://api.openai.com/v1", "ключ", False),
    ("https://8.8.8.8/v1", None, False),
    (None, "ключ", False),                              # пустой адрес — облако по умолчанию
    (None, None, True),                                 # модель не настроена
])
def test_model_endpoint_must_be_inside(base_url, api_key, allowed, monkeypatch):
    monkeypatch.delenv("ALLOW_EXTERNAL_MODELS", raising=False)
    monkeypatch.delenv("INTERNAL_HOSTS", raising=False)
    assert netguard.model_endpoint_allowed(base_url, api_key)[0] is allowed


def test_internal_server_by_name_and_explicit_permission(monkeypatch):
    monkeypatch.setenv("INTERNAL_HOSTS", "gpu.corp")
    assert netguard.model_endpoint_allowed("http://gpu.corp/v1", None)[0]
    assert not netguard.model_endpoint_allowed("http://gpu.example.com/v1", None)[0]
    monkeypatch.setenv("ALLOW_EXTERNAL_MODELS", "1")
    assert netguard.model_endpoint_allowed("https://api.openai.com/v1", "ключ")[0]


def _upload(client, headers, message, files):
    return client.post("/api/ask-files", data={"message": message}, headers=headers,
                       files=[("files", (name, content, mime)) for name, content, mime in files])


def test_employee_paths_stay_on_the_machine(client):
    headers = register(client, "offline@test.local")
    changed = list(LETTER)
    changed[2] = changed[2].replace("480 000", "520 000")
    with netguard.no_external_connections() as log:
        client.post("/api/ask", json={"question": "Кто согласует заявку на транспорт?"}, headers=headers)
        what = _upload(client, headers, "", [("письмо.docx", _docx(LETTER), DOCX)])
        compared = _upload(client, headers, "сравни, старая — первая",
                           [("a.docx", _docx(LETTER), DOCX), ("b.docx", _docx(changed), DOCX)])
    assert what.status_code == 200 and compared.status_code == 200
    assert "**480** 000" in compared.json()["answer"]
    assert log.blocked == []


@needs_ocr
def test_scan_registry_stays_on_the_machine(client):
    headers = register(client, "offline-scan@test.local")
    with netguard.no_external_connections() as log:
        resp = _upload(client, headers, "составь реестр спецификаций",
                       [("spec.pdf", _synthetic_scan(), "application/pdf")])
    assert resp.status_code == 200 and resp.json()["attachment"] is not None
    assert log.blocked == []
