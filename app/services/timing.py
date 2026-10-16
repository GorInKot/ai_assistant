"""Замер времени задач «вручную / с ассистентом» — основа расчёта эффекта.

Протокол: одна и та же задача (перенос листа спецификации в Excel, сравнение двух редакций,
поиск ответа) выполняется вручную и с ассистентом на разных, но сопоставимых документах.
Ведущий запускает и останавливает секундомер, затем вносит результат сверки с эталоном
(ошибки, пропуски). Машинное время набегает само: пока у ведущего идёт замер «с ассистентом»,
время, за которое ассистент обработал его запросы, прибавляется к замеру. Время человека —
всё время задачи минус машинное.

Сводка — медианы на единицу работы (лист, пара, вопрос) по задаче и режиму и экономия;
выгрузка CSV — для пересчёта в часы, ставки и рубли. Участник — код («У1»), не ФИО.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from statistics import median

from sqlalchemy.orm import Session

from app.db import TimingRun

TASKS = {
    "spec_to_excel": ("Перенос спецификации в Excel", "лист"),
    "compare_versions": ("Сравнение редакций", "пару"),  # «на пару»
    "search": ("Поиск ответа в документах", "вопрос"),
}
MODES = {"manual": "вручную", "assisted": "с ассистентом"}
DOC_TYPES = {"clean": "чистый скан", "handwritten": "скан с правками", "text": "текстовый документ", "": "—"}
STATUSES = {"running": "идёт", "done": "завершён", "failed": "сбой"}
RESULT_FIELDS = ("sheets", "positions", "flagged", "errors_wrong", "errors_missed", "errors_extra",
                 "changes_missed", "changes_false")
MAX_TEXT = 200
MAX_COMMENT = 2000


class TimingError(ValueError):
    """Ошибка ведущего: неизвестная задача, второй одновременный замер и т.п."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime | None) -> datetime | None:
    # SQLite возвращает время без пояснения пояса; пишем всегда UTC
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def active_run(db: Session, user_id: int) -> TimingRun | None:
    return (db.query(TimingRun)
            .filter(TimingRun.created_by == user_id, TimingRun.status == "running")
            .order_by(TimingRun.id.desc()).first())


def _text(value: str, field: str, required: bool = True) -> str:
    value = (value or "").strip()
    if required and not value:
        raise TimingError(f"Не заполнено поле «{field}»")
    return value[:MAX_TEXT]


def start(db: Session, user_id: int, *, participant: str, task: str, document: str,
          doc_type: str, mode: str) -> TimingRun:
    if task not in TASKS:
        raise TimingError("Неизвестная задача")
    if mode not in MODES:
        raise TimingError("Неизвестный режим")
    if doc_type not in DOC_TYPES:
        raise TimingError("Неизвестный тип документа")
    if active_run(db, user_id) is not None:
        raise TimingError("Уже идёт замер — сначала остановите его")
    run = TimingRun(created_by=user_id, participant=_text(participant, "Участник"), task=task,
                    document=_text(document, "Документ"), doc_type=doc_type, mode=mode,
                    status="running", started_at=_now(), machine_seconds=0.0)
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def stop(db: Session, run: TimingRun, *, failed: bool = False) -> TimingRun:
    if run.status != "running":
        raise TimingError("Замер уже остановлен")
    run.finished_at = _now()
    run.status = "failed" if failed else "done"
    db.commit()
    db.refresh(run)
    return run


def update(db: Session, run: TimingRun, fields: dict) -> TimingRun:
    """Результаты сверки с эталоном и комментарий. Пустое значение — «не считали»."""
    for name in RESULT_FIELDS:
        if name in fields:
            value = fields[name]
            if value is not None and (not isinstance(value, int) or value < 0):
                raise TimingError("Результаты — целые неотрицательные числа")
            setattr(run, name, value)
    if "comment" in fields:
        run.comment = (fields["comment"] or "").strip()[:MAX_COMMENT] or None
    db.commit()
    db.refresh(run)
    return run


def add_machine_seconds(db: Session, user_id: int, seconds: float) -> None:
    """Время обработки запроса ведущего — в идущий замер «с ассистентом», если он есть."""
    run = active_run(db, user_id)
    if run is not None and run.mode == "assisted" and seconds > 0:
        run.machine_seconds = (run.machine_seconds or 0.0) + seconds
        db.commit()


def total_seconds(run: TimingRun, now: datetime | None = None) -> float:
    end = _utc(run.finished_at) or (now or _now())
    return max((end - _utc(run.started_at)).total_seconds(), 0.0)


def human_seconds(run: TimingRun, now: datetime | None = None) -> float:
    return max(total_seconds(run, now) - (run.machine_seconds or 0.0), 0.0)


def units(run: TimingRun) -> int:
    """Сколько единиц работы в замере: листов для переноса спецификации, иначе одна."""
    return run.sheets if run.task == "spec_to_excel" and run.sheets else 1


def serialize(run: TimingRun) -> dict:
    return {
        "id": run.id,
        "participant": run.participant,
        "task": run.task,
        "document": run.document,
        "doc_type": run.doc_type,
        "mode": run.mode,
        "status": run.status,
        "started_at": _utc(run.started_at).isoformat(),
        "finished_at": _utc(run.finished_at).isoformat() if run.finished_at else None,
        "total_seconds": round(total_seconds(run), 1),
        "machine_seconds": round(run.machine_seconds or 0.0, 1),
        "human_seconds": round(human_seconds(run), 1),
        "comment": run.comment or "",
        **{name: getattr(run, name) for name in RESULT_FIELDS},
    }


def _stats(values: list[float]) -> dict:
    return {"n": len(values), "median": round(median(values), 1), "min": round(min(values), 1),
            "max": round(max(values), 1)}


def summary(runs: list[TimingRun]) -> list[dict]:
    """По задаче: медиана времени на единицу работы вручную и с ассистентом, экономия.
    Учитываются только завершённые замеры (не «сбой» и не идущие)."""
    out = []
    for task, (title, unit) in TASKS.items():
        row = {"task": task, "title": title, "unit": unit}
        for mode in MODES:
            done = [r for r in runs if r.task == task and r.mode == mode and r.status == "done"]
            row[mode] = _stats([total_seconds(r) / units(r) for r in done]) if done else None
        manual, assisted = row["manual"], row["assisted"]
        if manual and assisted and manual["median"] > 0:
            saved = manual["median"] - assisted["median"]
            row["saved_seconds"] = round(saved, 1)
            row["saved_share"] = round(saved / manual["median"], 3)
        else:
            row["saved_seconds"] = row["saved_share"] = None
        out.append(row)
    return out


CSV_COLUMNS = (
    ("id", "№"), ("participant", "участник"), ("task", "задача"), ("document", "документ"),
    ("doc_type", "тип документа"), ("mode", "режим"), ("status", "статус"), ("started_at", "старт (UTC)"),
    ("finished_at", "стоп (UTC)"), ("total_seconds", "всего, с"), ("machine_seconds", "машинное, с"),
    ("human_seconds", "человек, с"), ("sheets", "листов"), ("positions", "позиций"), ("flagged", "помечено"),
    ("errors_wrong", "неверных значений"), ("errors_missed", "пропущено позиций"),
    ("errors_extra", "лишних позиций"), ("changes_missed", "пропущено изменений"),
    ("changes_false", "ложных изменений"), ("comment", "комментарий"),
)
_LABELS = {"task": {k: v[0] for k, v in TASKS.items()}, "mode": MODES, "doc_type": DOC_TYPES, "status": STATUSES}


def _cell(name: str, value) -> str:
    if value is None:
        return ""
    if name in _LABELS:
        return _LABELS[name].get(value, str(value))
    if isinstance(value, float):
        return f"{value:.1f}".replace(".", ",")  # русский Excel ждёт запятую
    text = str(value)
    # текст, похожий на формулу, Excel не должен исполнять
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def export_csv(runs: list[TimingRun]) -> bytes:
    """CSV для Excel: точка с запятой, BOM — кириллица открывается без настроек."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";")
    writer.writerow([title for _, title in CSV_COLUMNS])
    for run in runs:
        data = serialize(run)
        writer.writerow([_cell(name, data[name]) for name, _ in CSV_COLUMNS])
    return ("﻿" + buf.getvalue()).encode("utf-8")
