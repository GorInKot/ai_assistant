"""Кнопка ассистента на всех страницах портала без правки его файлов.

ASGI-обёртка над всем приложением портала (Django и СИЗ — см. portalrnsk/asgi.py) дописывает
`<script ... widget.js>` перед `</body>` в HTML-ответы для вошедших пользователей. Только
стандартная библиотека — модуль проверяется тестами ассистента без Django; что такое «вошедший»
и какой тег вставлять, передаёт portal_bridge/asgi.py.

Не трогает: не-GET, ответы не 200, не text/html, сжатые, фрагменты без `</body>`, страницы, где
кнопка уже есть (атрибут data-ai-assistant-widget — например, подключена строкой в base.html),
и страницы из EXCLUDED_PATHS (вход и сам ассистент).
"""

from __future__ import annotations

from html import escape
from typing import Awaitable, Callable

MARKER = b"data-ai-assistant-widget"
EXCLUDED_PATHS = ("/login/", "/logout/", "/register/", "/assistant/", "/assistant/panel/")


def script_tag(assistant_url: str, panel_url: str, full_url: str) -> bytes:
    src = escape(f"{assistant_url.rstrip('/')}/widget.js")
    return (
        f'<script src="{src}" data-panel="{escape(panel_url)}" data-full="{escape(full_url)}" '
        f"data-ai-assistant-widget defer></script>\n"
    ).encode("utf-8")


def insert_script(html: bytes, tag: bytes) -> bytes | None:
    """Страница с тегом перед последним </body>; None — вставлять не нужно."""
    if MARKER in html:
        return None
    index = html.lower().rfind(b"</body>")
    if index == -1:
        return None
    return html[:index] + tag + html[index:]


def _header(headers, name: bytes) -> bytes:
    for key, value in headers:
        if key.lower() == name:
            return value
    return b""


class WidgetInjector:
    def __init__(
        self,
        app,
        *,
        is_authenticated: Callable[[dict], Awaitable[bool]],
        tag_for: Callable[[dict], bytes],
    ):
        self.app = app
        self.is_authenticated = is_authenticated
        self.tag_for = tag_for

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "GET"
            or scope.get("path") in EXCLUDED_PATHS
            or not await self.is_authenticated(scope)
        ):
            await self.app(scope, receive, send)
            return

        start: dict | None = None
        chunks: list[bytes] = []

        async def send_with_widget(message):
            nonlocal start
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                html = _header(headers, b"content-type").lower().startswith(b"text/html")
                if message.get("status") == 200 and html and not _header(headers, b"content-encoding"):
                    start = message  # тело соберём целиком и отправим вместе с заголовками
                    return
                await send(message)
                return
            if message["type"] != "http.response.body" or start is None:
                await send(message)
                return
            chunks.append(message.get("body", b""))
            if message.get("more_body"):
                return
            body = b"".join(chunks)
            body = insert_script(body, self.tag_for(scope)) or body
            headers = [(k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"]
            headers.append((b"content-length", str(len(body)).encode("ascii")))
            await send({**start, "headers": headers})
            await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, send_with_widget)
