"""Адреса портала как есть плюс /assistant/, /assistant/tools/ и /assistant/panel/ (панель кнопки);
главная — с плитками ассистента."""

from django.urls import path

from portalrnsk.urls import urlpatterns as portal_urlpatterns

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("assistant/", views.assistant, name="assistant"),
    path("assistant/tools/", views.assistant_tools, name="assistant_tools"),
    path("assistant/panel/", views.assistant_panel, name="assistant_panel"),
    *portal_urlpatterns,
]
