"""Адреса портала как есть плюс /assistant/, /assistant/tools/ и /assistant/panel/ (панель кнопки);
главная — с плитками ассистента."""

from django.urls import path

from portalrnsk.urls import urlpatterns as portal_urlpatterns

from . import meeting, views

urlpatterns = [
    path("", views.home, name="home"),
    path("assistant/", views.assistant, name="assistant"),
    path("assistant/tools/", views.assistant_tools, name="assistant_tools"),
    path("assistant/panel/", views.assistant_panel, name="assistant_panel"),
    path("protocol/", meeting.protocol_home, name="meeting_protocol_home"),
    path("protocol/api/state", meeting.protocol_state_with_sources, name="meeting_protocol_state"),
    path("protocol/meeting/", meeting.meeting_page, name="meeting_page"),
    path("protocol/meeting/upload/", meeting.meeting_upload, name="meeting_upload"),
    path("protocol/meeting/<str:job_id>/", meeting.meeting_status, name="meeting_status"),
    path("protocol/meeting/<str:job_id>/confirm/", meeting.meeting_confirm, name="meeting_confirm"),
    path("protocol/meeting/<str:job_id>/media/", meeting.meeting_media, name="meeting_media"),
    path("tasks/", meeting.task_board, name="task_board"),
    *portal_urlpatterns,
]
