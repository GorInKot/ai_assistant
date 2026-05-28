from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock


CLARIFICATION_HINT_RE = re.compile(
    r"^(по|про|в|для)\s+|^(ектп|цус|обучение|медосмотр|transport|training)\b",
    re.IGNORECASE,
)


@dataclass
class PendingClarification:
    """Незаконченный QA-вопрос, ждём уточнения от пользователя."""

    base_question: str
    created_at: datetime


@dataclass
class PendingRequest:
    """Незаконченная заявка: ассистент опрашивает слоты по одному.

    Когда все required-слоты собраны и пользователь подтвердил —
    создаём Request в БД. Состояние сбрасывается по TTL или явной отмене.
    """

    request_type: str
    filled_slots: dict[str, str] = field(default_factory=dict)
    awaiting_slot: str | None = None
    awaiting_confirmation: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DialogStateStore:
    """In-memory state на сессию (per-user через session_id).

    Хранит ОДНО из двух состояний на сессию:
    - pending clarification (для QA);
    - pending request (для slot-filling заявки).
    Активное состояние подменяет старое; explicit clear() сбрасывает всё.
    """

    def __init__(self, ttl_minutes: int = 20) -> None:
        self.ttl = timedelta(minutes=ttl_minutes)
        self._clarifications: dict[str, PendingClarification] = {}
        self._requests: dict[str, PendingRequest] = {}
        self._lock = Lock()

    # ---- Clarifications (QA flow) ----

    def merge_with_pending(self, session_id: str, incoming_question: str) -> tuple[str, bool]:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._cleanup(now)
            pending = self._clarifications.get(session_id)
            if not pending:
                return incoming_question, False

            if self._looks_like_clarification(incoming_question):
                merged = f"{pending.base_question}\nУточнение пользователя: {incoming_question}"
                return merged, True

            # User asked a new full question; drop stale clarification context.
            self._clarifications.pop(session_id, None)
            return incoming_question, False

    def set_pending(self, session_id: str, base_question: str) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._cleanup(now)
            self._clarifications[session_id] = PendingClarification(
                base_question=base_question,
                created_at=now,
            )

    # ---- Pending requests (slot-filling) ----

    def get_pending_request(self, session_id: str) -> PendingRequest | None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._cleanup(now)
            return self._requests.get(session_id)

    def set_pending_request(self, session_id: str, request: PendingRequest) -> None:
        with self._lock:
            self._cleanup(datetime.now(timezone.utc))
            self._requests[session_id] = request
            # Активная заявка отменяет clarification-контекст — иначе можно
            # получить мерж QA-вопроса в текст слот-ответа.
            self._clarifications.pop(session_id, None)

    def clear_pending_request(self, session_id: str) -> None:
        with self._lock:
            self._requests.pop(session_id, None)

    # ---- Common ----

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._clarifications.pop(session_id, None)
            self._requests.pop(session_id, None)

    def _looks_like_clarification(self, text: str) -> bool:
        value = text.strip()
        if not value:
            return False

        token_count = len(value.split())
        if token_count <= 4:
            return True

        if "?" in value:
            return False

        if token_count <= 8 and CLARIFICATION_HINT_RE.search(value):
            return True

        return False

    def _cleanup(self, now: datetime) -> None:
        stale_clar = [sid for sid, st in self._clarifications.items() if now - st.created_at > self.ttl]
        for sid in stale_clar:
            self._clarifications.pop(sid, None)
        stale_req = [sid for sid, st in self._requests.items() if now - st.created_at > self.ttl]
        for sid in stale_req:
            self._requests.pop(sid, None)
