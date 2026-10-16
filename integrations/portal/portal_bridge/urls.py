"""Адреса портала как есть плюс /assistant/; главная — с плиткой ассистента."""

from django.urls import path

from portalrnsk.urls import urlpatterns as portal_urlpatterns

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("assistant/", views.assistant, name="assistant"),
    *portal_urlpatterns,
]
