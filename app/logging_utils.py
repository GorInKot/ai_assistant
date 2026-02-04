from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RequestLogger:
    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, query: str, sources: list[dict[str, Any]], answer: str | None = None) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_query": query,
            "selected_sources": [
                {
                    "file_name": source.get("file_name"),
                    "relative_path": source.get("relative_path"),
                    "pages": source.get("pages", []),
                    "sections": source.get("sections", []),
                    "source_type": source.get("source_type", "context"),
                }
                for source in sources
            ],
        }
        if answer is not None:
            payload["answer"] = answer

        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
