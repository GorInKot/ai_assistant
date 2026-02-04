from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ActionRecord:
    action_id: str
    action_type: str
    process: str
    title: str
    details: str
    requester: str
    status: str
    created_at: str


class ActionsStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("", encoding="utf-8")

    def create_action(
        self,
        action_type: str,
        process: str,
        title: str,
        details: str,
        requester: str,
    ) -> ActionRecord:
        record = ActionRecord(
            action_id=f"ACT-{uuid4().hex[:8].upper()}",
            action_type=action_type,
            process=process,
            title=title,
            details=details,
            requester=requester or "Не указан",
            status="Черновик (локально)",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._append_record(record)
        return record

    def list_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            return []

        lines = self.file_path.read_text(encoding="utf-8").splitlines()
        items: list[dict[str, Any]] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(items) >= limit:
                break
        return items

    def _append_record(self, record: ActionRecord) -> None:
        payload = {
            "action_id": record.action_id,
            "action_type": record.action_type,
            "process": record.process,
            "title": record.title,
            "details": record.details,
            "requester": record.requester,
            "status": record.status,
            "created_at": record.created_at,
            "integration_note": "MVP: действие зарегистрировано локально, в корпоративные ИС не отправляется.",
        }
        with self.file_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
