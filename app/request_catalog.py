"""Каталог типов заявок — кэш в памяти, source of truth = БД.

До security/admin-фазы каталог жил в app/data/request_types.yaml. Теперь
YAML — initial seed, а edits идут через /api/admin/request-types и пишутся
в таблицы request_types/request_type_slots (см. app/db.py).

Используем in-memory snapshot (dict[slug, RequestTypeDef]), который
перестраивается reload_catalog() после любого CRUD-действия. Чтение
get_request_type/all_request_types бесплатное и не блокирует БД.

Старт: модуль импортируется до того, как FastAPI создал app — поэтому
загрузка идёт лениво (при первом обращении), а init_db() из app/main.py
успевает наполнить таблицу.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from threading import Lock

from app.db import RequestType, RequestTypeSlot, SessionLocal


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlotDef:
    name: str
    question: str
    required: bool


@dataclass(frozen=True)
class RequestTypeDef:
    type: str
    title: str
    responsibility_area: str
    is_anonymous: bool
    trigger_keywords: tuple[str, ...]
    examples: tuple[str, ...]
    slots: tuple[SlotDef, ...]

    def required_slot_names(self) -> list[str]:
        return [s.name for s in self.slots if s.required]


_CACHE: dict[str, RequestTypeDef] | None = None
_CACHE_LOCK = Lock()


def _load_from_db() -> dict[str, RequestTypeDef]:
    db = SessionLocal()
    try:
        rows = (
            db.query(RequestType)
            .filter(RequestType.is_active.is_(True))
            .order_by(RequestType.sort_order, RequestType.id)
            .all()
        )
        catalog: dict[str, RequestTypeDef] = {}
        for row in rows:
            try:
                triggers = json.loads(row.trigger_keywords_json or "[]")
                examples = json.loads(row.examples_json or "[]")
            except json.JSONDecodeError:
                triggers, examples = [], []
            slots = tuple(
                SlotDef(name=s.name, question=s.question, required=bool(s.required))
                for s in row.slots
            )
            catalog[row.type_slug] = RequestTypeDef(
                type=row.type_slug,
                title=row.title,
                responsibility_area=row.responsibility_area_slug,
                is_anonymous=bool(row.is_anonymous),
                trigger_keywords=tuple(str(t) for t in triggers),
                examples=tuple(str(e) for e in examples),
                slots=slots,
            )
        return catalog
    finally:
        db.close()


def reload_catalog() -> dict[str, RequestTypeDef]:
    """Принудительно перечитать каталог из БД (вызывается после CRUD)."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = _load_from_db()
        return _CACHE


def _ensure_loaded() -> dict[str, RequestTypeDef]:
    global _CACHE
    if _CACHE is None:
        with _CACHE_LOCK:
            if _CACHE is None:
                _CACHE = _load_from_db()
    return _CACHE


def get_request_type(type_slug: str) -> RequestTypeDef | None:
    return _ensure_loaded().get(type_slug)


def all_request_types() -> list[RequestTypeDef]:
    return list(_ensure_loaded().values())
