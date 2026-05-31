"""Тесты email-нотификатора заявок (офлайн, smtplib замокан)."""

from __future__ import annotations

import app.notifications as notifications
from app.notifications import EmailNotifier, SMTPConfig


DISABLED = SMTPConfig(host=None, port=587, user=None, password=None, sender=None, use_tls=True)
ENABLED = SMTPConfig(
    host="smtp.example.com",
    port=587,
    user="bot@example.com",
    password="secret",
    sender="bot@example.com",
    use_tls=True,
)


def _notify(notifier: EmailNotifier, **overrides):
    payload = dict(
        to_email="manager@corp.ru",
        employee_name="Пётр Петров",
        request_id=42,
        type_title="Заявка на транспорт",
        summary="Заявка на транспорт: завтра",
        is_anonymous=False,
        requester_name="Иван Сидоров",
    )
    payload.update(overrides)
    return notifier.notify_new_request(**payload)


class _FakeSMTP:
    """Подмена smtplib.SMTP: записывает отправленные сообщения и вызовы."""

    sent: list = []
    started_tls = False
    logged_in = False

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        type(self).started_tls = True

    def login(self, user, password):
        type(self).logged_in = True

    def send_message(self, msg):
        type(self).sent.append(msg)


def test_disabled_notifier_is_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(notifications.smtplib, "SMTP", lambda *a, **k: calls.append(1))
    notifier = EmailNotifier(DISABLED)
    assert notifier.enabled is False
    assert _notify(notifier) is False
    assert calls == []  # к SMTP вообще не обращались


def test_no_recipient_is_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(notifications.smtplib, "SMTP", lambda *a, **k: calls.append(1))
    notifier = EmailNotifier(ENABLED)
    assert _notify(notifier, to_email=None) is False
    assert calls == []


def test_enabled_notifier_sends(monkeypatch):
    _FakeSMTP.sent = []
    _FakeSMTP.started_tls = False
    _FakeSMTP.logged_in = False
    monkeypatch.setattr(notifications.smtplib, "SMTP", _FakeSMTP)

    notifier = EmailNotifier(ENABLED)
    assert _notify(notifier) is True
    assert len(_FakeSMTP.sent) == 1
    msg = _FakeSMTP.sent[0]
    assert msg["To"] == "manager@corp.ru"
    assert "#42" in msg["Subject"]
    assert _FakeSMTP.started_tls is True
    assert _FakeSMTP.logged_in is True
    # Не анонимная заявка → инициатор раскрыт в теле.
    assert "Иван Сидоров" in msg.get_content()


def test_anonymous_hides_requester_in_email(monkeypatch):
    _FakeSMTP.sent = []
    monkeypatch.setattr(notifications.smtplib, "SMTP", _FakeSMTP)

    notifier = EmailNotifier(ENABLED)
    assert _notify(notifier, is_anonymous=True, requester_name="Иван Сидоров") is True
    body = _FakeSMTP.sent[-1].get_content()
    assert "Иван Сидоров" not in body
    assert "анонимн" in body.lower()


def test_send_failure_does_not_raise(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(notifications.smtplib, "SMTP", boom)
    notifier = EmailNotifier(ENABLED)
    # Сбой отправки не пробрасывается — создание заявки не должно падать.
    assert _notify(notifier) is False
