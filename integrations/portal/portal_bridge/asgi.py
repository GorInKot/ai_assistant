"""Приложение портала как есть (Django + СИЗ из portalrnsk/asgi.py) с кнопкой ассистента
на всех страницах для вошедших пользователей (widget_inject.py).

Запуск: uvicorn portal_bridge.asgi:application (см. run_portal.sh).
"""

from __future__ import annotations

import os
from http.cookies import SimpleCookie
from importlib import import_module

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "portal_bridge.settings")

from portalrnsk.asgi import application as portal_application  # noqa: E402  (настраивает Django)

from asgiref.sync import sync_to_async  # noqa: E402
from django.conf import settings  # noqa: E402
from django.urls import reverse  # noqa: E402

from .views import MIN_SECRET_LEN, assistant_base_url  # noqa: E402
from .widget_inject import WidgetInjector, script_tag  # noqa: E402


def _header(scope, name: bytes) -> str:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin1")
    return ""


@sync_to_async
def _session_user_is_active(session_key: str) -> bool:
    from django.contrib.auth import SESSION_KEY, get_user_model

    # SessionStore сам отбрасывает истёкшие и подделанные сессии.
    user_id = import_module(settings.SESSION_ENGINE).SessionStore(session_key).get(SESSION_KEY)
    return bool(user_id) and get_user_model().objects.filter(pk=user_id, is_active=True).exists()


async def is_authenticated(scope) -> bool:
    if not settings.ASSISTANT_WIDGET or len(settings.ASSISTANT_SSO_SECRET) < MIN_SECRET_LEN:
        return False  # без секрета панель не войдёт — кнопку не показываем
    cookies = SimpleCookie()
    try:
        cookies.load(_header(scope, b"cookie"))
    except Exception:
        return False
    morsel = cookies.get(settings.SESSION_COOKIE_NAME)
    return bool(morsel and morsel.value) and await _session_user_is_active(morsel.value)


def tag_for(scope) -> bytes:
    url = assistant_base_url(scope.get("scheme", "http"), _header(scope, b"host"))
    return script_tag(url, reverse("assistant_panel"), reverse("assistant"))


application = WidgetInjector(portal_application, is_authenticated=is_authenticated, tag_for=tag_for)
