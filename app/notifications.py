"""Email-уведомления о новых заявках (опционально, env-driven).

Включается, только если заданы SMTP_HOST и SMTP_FROM — иначе тихий no-op
(тот же приём, что у LLM/embedder: фича готова, но не мешает, пока не настроена).
Сбой отправки НИКОГДА не должен ломать создание заявки — поэтому всё письмо
оборачивается в try/except и логируется.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SMTPConfig:
    host: str | None
    port: int
    user: str | None
    password: str | None
    sender: str | None
    use_tls: bool


class EmailNotifier:
    def __init__(self, config: SMTPConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.host and self.config.sender)

    def notify_new_request(
        self,
        *,
        to_email: str | None,
        employee_name: str,
        request_id: int,
        type_title: str,
        summary: str | None,
        is_anonymous: bool,
        requester_name: str | None = None,
    ) -> bool:
        """Шлёт ответственному письмо о назначенной заявке. Возвращает True,
        если письмо ушло. Никогда не бросает — ошибки только логируются."""
        if not self.enabled or not to_email:
            return False

        try:
            msg = EmailMessage()
            msg["Subject"] = f"Новая заявка #{request_id}: {type_title}"
            msg["From"] = self.config.sender
            msg["To"] = to_email

            lines = [
                f"Здравствуйте, {employee_name}!",
                "",
                f"На вас назначена новая заявка #{request_id} «{type_title}».",
            ]
            if summary:
                lines.append(f"Кратко: {summary}")
            if is_anonymous:
                lines.append("Обращение анонимное — отправитель скрыт.")
            elif requester_name:
                lines.append(f"Инициатор: {requester_name}")
            lines += [
                "",
                "Откройте раздел «Кабинет блока» в ассистенте, чтобы обработать заявку.",
            ]
            msg.set_content("\n".join(lines))

            self._send(msg)
            logger.info("Email-уведомление о заявке #%s отправлено на %s", request_id, to_email)
            return True
        except Exception:
            logger.exception("Не удалось отправить email-уведомление о заявке #%s", request_id)
            return False

    def _send(self, msg: EmailMessage) -> None:
        cfg = self.config
        with smtplib.SMTP(cfg.host, cfg.port, timeout=10) as server:
            if cfg.use_tls:
                server.starttls(context=ssl.create_default_context())
            if cfg.user and cfg.password:
                server.login(cfg.user, cfg.password)
            server.send_message(msg)
