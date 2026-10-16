"""Кнопка ассистента на страницах портала: вставка widget.js надстройкой и вход панели."""

from __future__ import annotations

import asyncio
import re
import secrets
import sys
import time
from pathlib import Path

import pytest

from app.routers import portal_sso

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integrations" / "portal"))
try:
    from portal_bridge.widget_inject import WidgetInjector, insert_script, script_tag
finally:
    sys.path.pop(0)

TAG = script_tag("http://assistant:8000", "/assistant/panel/", "/assistant/")
PAGE = b"<html><body><h1>\xd0\x9c\xd0\xb5\xd1\x82\xd1\x80\xd0\xbe\xd0\xbb\xd0\xbe\xd0\xb3\xd0\xb8\xd1\x8f</h1></body></html>"


def _app(body: bytes = PAGE, content_type: bytes = b"text/html; charset=utf-8", status: int = 200,
         chunks: int = 1, extra_headers: list | None = None):
    async def app(scope, receive, send):
        headers = [(b"content-type", content_type), (b"content-length", str(len(body)).encode())]
        await send({"type": "http.response.start", "status": status, "headers": headers + (extra_headers or [])})
        step = max(1, len(body) // chunks)
        parts = [body[i:i + step] for i in range(0, len(body), step)] or [b""]
        for i, part in enumerate(parts):
            await send({"type": "http.response.body", "body": part, "more_body": i < len(parts) - 1})
    return app


def _run(app, *, path="/metrology/", method="GET", logged_in=True):
    async def is_auth(scope):
        return logged_in

    injector = WidgetInjector(app, is_authenticated=is_auth, tag_for=lambda scope: TAG)
    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": method, "path": path, "headers": []}
    asyncio.run(injector(scope, None, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start, body


def test_script_goes_before_body_end_for_logged_in_user():
    start, body = _run(_app())
    assert body.count(b"data-ai-assistant-widget") == 1
    assert body.index(b"widget.js") < body.index(b"</body>")
    assert dict(start["headers"])[b"content-length"] == str(len(body)).encode()


def test_html_sent_in_parts_is_collected():
    _start, body = _run(_app(chunks=5))
    assert body.startswith(b"<html>") and b"widget.js" in body


@pytest.mark.parametrize("kwargs", [
    {"content_type": b"application/json"},
    {"content_type": b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    {"status": 302},
    {"status": 404},
    {"extra_headers": [(b"content-encoding", b"gzip")]},
    {"body": b"<tr><td>fragment</td></tr>"},
])
def test_non_pages_are_untouched(kwargs):
    body_in = kwargs.get("body", PAGE)
    _start, body = _run(_app(**kwargs))
    assert body == body_in


def test_only_for_logged_in_users_and_get():
    assert b"widget.js" not in _run(_app(), logged_in=False)[1]
    assert b"widget.js" not in _run(_app(), method="POST")[1]


@pytest.mark.parametrize("path", ["/login/", "/assistant/", "/assistant/panel/"])
def test_no_button_on_login_and_assistant_pages(path):
    assert b"widget.js" not in _run(_app(), path=path)[1]


def test_button_on_siz_and_tools_pages():
    assert b"widget.js" in _run(_app(), path="/siz/")[1]
    assert b"widget.js" in _run(_app(), path="/assistant/tools/")[1]


def test_no_second_button_when_already_connected():
    page = PAGE.replace(b"</body>", TAG + b"</body>")
    assert insert_script(page, TAG) is None
    assert _run(_app(body=page))[1].count(b"data-ai-assistant-widget") == 1


def test_tag_attributes_are_escaped():
    tag = script_tag('http://a"><script>x</script>', "/p/", "/f/")
    assert b"<script>x" not in tag and b"&quot;" in tag


# ---------- вход панели ----------

SECRET = "s" * 40


@pytest.fixture(autouse=True)
def _sso_env(monkeypatch):
    monkeypatch.setenv("PORTAL_SSO_SECRET", SECRET)
    monkeypatch.setenv("PORTAL_ORIGINS", "http://127.0.0.1:8080 http://localhost:8080")


def _pass() -> str:
    return portal_sso.sign({"sub": "ivanov", "role": "user", "exp": int(time.time()) + 60,
                            "nonce": secrets.token_hex(16)}, SECRET)


def _landing(client, **form):
    resp = client.post("/sso/portal", data={"token": _pass(), **form})
    assert resp.status_code == 200, resp.text
    target = re.search(r"location\.replace\((\"[^\"]*\")\)", resp.text).group(1)
    return target, resp.text


def test_panel_login_goes_to_widget_and_remembers_portal(client):
    target, html = _landing(client, next="/widget", host="http://127.0.0.1:8080")
    assert target == '"/widget"'
    assert 'sessionStorage.setItem("ai_assistant_host", "http://127.0.0.1:8080")' in html


@pytest.mark.parametrize("host", ["https://evil.example", "http://127.0.0.1:8080.evil", "", "null"])
def test_unknown_host_is_not_remembered(client, host):
    _target, html = _landing(client, next="/widget", host=host)
    assert "setItem(\"ai_assistant_host\"" not in html and 'removeItem("ai_assistant_host")' in html


def test_open_full_leads_to_same_conversation(client):
    assert _landing(client, conversation="42")[0] == '"/?conversation=42"'
    assert _landing(client, next="/tools", conversation="42")[0] == '"/tools"'
    for bad in ("0", "-1", "4a", "1" * 13, "</script>"):
        assert _landing(client, conversation=bad)[0] == '"/"'


def test_widget_script_is_served(client):
    resp = client.get("/widget.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    assert "event.origin !== assistantOrigin" in resp.text
