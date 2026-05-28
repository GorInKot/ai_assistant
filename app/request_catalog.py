"""Загрузчик каталога типов заявок из YAML.

Загружается один раз при импорте модуля. Если потребуется горячая перезагрузка
без рестарта — можно добавить reload() аналогично KB.reindex().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml


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


_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "request_types.yaml"


def _load_catalog(path: Path) -> dict[str, RequestTypeDef]:
    if not path.exists():
        logger.warning("Request types catalog not found: %s", path)
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    types_raw = raw.get("request_types", [])
    catalog: dict[str, RequestTypeDef] = {}

    for item in types_raw:
        try:
            slots = tuple(
                SlotDef(
                    name=str(s["name"]),
                    question=str(s["question"]),
                    required=bool(s.get("required", False)),
                )
                for s in item.get("slots", [])
            )
            type_def = RequestTypeDef(
                type=str(item["type"]),
                title=str(item["title"]),
                responsibility_area=str(item["responsibility_area"]),
                is_anonymous=bool(item.get("is_anonymous", False)),
                trigger_keywords=tuple(item.get("trigger_keywords", [])),
                examples=tuple(item.get("examples", [])),
                slots=slots,
            )
            catalog[type_def.type] = type_def
        except (KeyError, TypeError) as err:
            logger.warning("Skipped malformed request type: %s (%s)", item, err)

    return catalog


REQUEST_TYPES: dict[str, RequestTypeDef] = _load_catalog(_CATALOG_PATH)


def get_request_type(type_slug: str) -> RequestTypeDef | None:
    return REQUEST_TYPES.get(type_slug)


def all_request_types() -> list[RequestTypeDef]:
    return list(REQUEST_TYPES.values())
