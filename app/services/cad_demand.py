"""Спрос на конструктор деталей: что просили построить, но это пока не поддерживается.

Источник — история запросов конструктора (`cad_requests`). В отчёт попадают:
- отказы «пока не поддерживается» (модель распознала, что запрошенного предмета конструктор не умеет);
- функции, которые пропущены при частичном построении (фаски, резьба…): деталь построена, но без них.

Формулировки модели различаются («тройники и ответвления» / «ответвления труб»), поэтому похожие
объединяются в темы тем же способом, что и вопросы в «Пробелах в знаниях». Считаются разные
сотрудники; имена не раскрываются, показывается только подразделение. Отчёт — сигнал для
владельца продукта, что добавить в конструктор следующим.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db import CadRequest, User
from app.services import cad, gaps
from app.services.cad_history import (
    OUTCOME_BUILT,
    OUTCOME_CLARIFY,
    OUTCOME_ERROR,
    OUTCOME_UNSUPPORTED,
)

MAX_EXAMPLES = 4
MAX_ITEMS = 500  # самые свежие; ограничивает и время запроса к эмбеддеру

KIND_REFUSED = "refused"  # конструктор отказал
KIND_PARTIAL = "partial"  # деталь построена без этой функции


def _split_top_level(text: str) -> list[str]:
    """Разбить по запятым вне скобок: «габариты 800 мм (не учитывается, только стык)» остаётся одной частью."""
    parts: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return parts


def _skipped_features(notes_json: str | None) -> list[str]:
    """Функции из пометки «Пока не поддерживается и не учтено: фаски, резьба.» (без мусора и размеров)."""
    try:
        notes = json.loads(notes_json or "[]")
    except json.JSONDecodeError:
        return []
    features: list[str] = []
    for note in notes:
        if isinstance(note, str) and note.startswith(cad.SKIPPED_NOTE_PREFIX):
            tail = note[len(cad.SKIPPED_NOTE_PREFIX):].rstrip(". ")
            features += cad.clean_skipped_features(_split_top_level(tail))
    return features


def build(db: Session, *, days: int = 30, embedder: Any = None) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = (
        db.query(CadRequest, User.division)
        .join(User, User.id == CadRequest.user_id)
        .filter(CadRequest.created_at >= cutoff)
        .order_by(CadRequest.id.desc())
        .all()
    )

    counts = {OUTCOME_BUILT: 0, OUTCOME_CLARIFY: 0, OUTCOME_UNSUPPORTED: 0, OUTCOME_ERROR: 0}
    items: list[dict] = []  # что именно не поддерживается, от свежих к старым
    for request, division in rows:
        counts[request.outcome] = counts.get(request.outcome, 0) + 1
        base = {"id": request.id, "description": request.description, "user_id": request.user_id,
                "division": division, "created_at": request.created_at}
        if request.outcome == OUTCOME_UNSUPPORTED and request.detail:
            items.append({**base, "what": request.detail, "kind": KIND_REFUSED})
        elif request.outcome == OUTCOME_BUILT:
            items += [{**base, "what": feature, "kind": KIND_PARTIAL}
                      for feature in _skipped_features(request.notes_json)]
    items = items[:MAX_ITEMS]

    clusters, used_embeddings = gaps.cluster_questions([i["what"] for i in items], embedder)
    topics: list[dict] = []
    for members in clusters:
        group = [items[i] for i in members]
        examples: list[str] = []
        for item in group:
            if item["description"] not in examples:
                examples.append(item["description"])
        created = group[0]["created_at"]
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        topics.append({
            "topic": group[0]["what"],  # формулировка самого свежего запроса
            "kind": KIND_REFUSED if any(i["kind"] == KIND_REFUSED for i in group) else KIND_PARTIAL,
            "requests": len({i["id"] for i in group}),
            "people": len({i["user_id"] for i in group}),
            "divisions": sorted({i["division"] for i in group if i["division"]}),
            "refused": sum(1 for i in group if i["kind"] == KIND_REFUSED),
            "partial": sum(1 for i in group if i["kind"] == KIND_PARTIAL),
            "examples": examples[:MAX_EXAMPLES],
            "last_asked": created.isoformat() if created else None,
        })
    topics.sort(key=lambda t: (-t["people"], -t["requests"], t["topic"]))

    return {
        "requests": len(rows),
        "built": counts[OUTCOME_BUILT],
        "unsupported": counts[OUTCOME_UNSUPPORTED],
        "clarify": counts[OUTCOME_CLARIFY],
        "errors": counts[OUTCOME_ERROR],
        "topics": topics,
        "grouping": "embeddings" if used_embeddings else "words",
    }
