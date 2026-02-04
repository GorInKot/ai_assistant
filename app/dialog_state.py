from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock


CLARIFICATION_HINT_RE = re.compile(
    r"^(по|про|в|для)\s+|^(ектп|цус|обучение|медосмотр|transport|training)\b",
    re.IGNORECASE,
)


@dataclass
class PendingClarification:
    base_question: str
    created_at: datetime


class DialogStateStore:
    def __init__(self, ttl_minutes: int = 20) -> None:
        self.ttl = timedelta(minutes=ttl_minutes)
        self._items: dict[str, PendingClarification] = {}
        self._lock = Lock()

    def merge_with_pending(self, session_id: str, incoming_question: str) -> tuple[str, bool]:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._cleanup(now)
            pending = self._items.get(session_id)
            if not pending:
                return incoming_question, False

            if self._looks_like_clarification(incoming_question):
                merged = f"{pending.base_question}\nУточнение пользователя: {incoming_question}"
                return merged, True

            # User asked a new full question; drop stale clarification context.
            self._items.pop(session_id, None)
            return incoming_question, False

    def set_pending(self, session_id: str, base_question: str) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._cleanup(now)
            self._items[session_id] = PendingClarification(
                base_question=base_question,
                created_at=now,
            )

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._items.pop(session_id, None)

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
        stale_sessions = [
            session_id
            for session_id, state in self._items.items()
            if now - state.created_at > self.ttl
        ]
        for session_id in stale_sessions:
            self._items.pop(session_id, None)
