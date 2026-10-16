"""Страница «Замер»: секундомер задач «вручную / с ассистентом» для расчёта эффекта (admin, manager).

Логика — services/timing.py. Каждый ведущий видит свои замеры.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_db, require_role
from app.db import ROLE_ADMIN, ROLE_MANAGER, TimingRun, User
from app.services import timing

router = APIRouter(prefix="/api/admin/timing")

admin_or_manager = require_role(ROLE_ADMIN, ROLE_MANAGER)


class StartRequest(BaseModel):
    participant: str = Field(..., max_length=200)
    task: str
    document: str = Field(..., max_length=200)
    doc_type: str = ""
    mode: str


class StopRequest(BaseModel):
    failed: bool = False


class ResultsUpdate(BaseModel):
    sheets: int | None = None
    positions: int | None = None
    flagged: int | None = None
    errors_wrong: int | None = None
    errors_missed: int | None = None
    errors_extra: int | None = None
    changes_missed: int | None = None
    changes_false: int | None = None
    comment: str | None = Field(default=None, max_length=timing.MAX_COMMENT)


def _runs(db: Session, user: User) -> list[TimingRun]:
    return db.query(TimingRun).filter(TimingRun.created_by == user.id).order_by(TimingRun.id.desc()).all()


def _own(db: Session, user: User, run_id: int) -> TimingRun:
    run = db.query(TimingRun).filter(TimingRun.id == run_id, TimingRun.created_by == user.id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Замер не найден")
    return run


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except timing.TimingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def overview(current_user: User = Depends(admin_or_manager), db: Session = Depends(get_db)) -> dict:
    runs = _runs(db, current_user)
    active = timing.active_run(db, current_user.id)
    return {
        "runs": [timing.serialize(r) for r in runs],
        "active": timing.serialize(active) if active else None,
        "summary": timing.summary(runs),
        "options": {
            "tasks": [{"value": k, "label": v[0], "unit": v[1]} for k, v in timing.TASKS.items()],
            "modes": [{"value": k, "label": v} for k, v in timing.MODES.items()],
            "doc_types": [{"value": k, "label": v} for k, v in timing.DOC_TYPES.items()],
        },
    }


@router.get("/active")
def active(current_user: User = Depends(admin_or_manager), db: Session = Depends(get_db)) -> dict:
    run = timing.active_run(db, current_user.id)
    return {"active": timing.serialize(run) if run else None}


@router.post("/runs")
def start(payload: StartRequest, current_user: User = Depends(admin_or_manager),
          db: Session = Depends(get_db)) -> dict:
    run = _guard(timing.start, db, current_user.id, **payload.model_dump())
    return timing.serialize(run)


@router.post("/runs/{run_id}/stop")
def stop(run_id: int, payload: StopRequest, current_user: User = Depends(admin_or_manager),
         db: Session = Depends(get_db)) -> dict:
    run = _guard(timing.stop, db, _own(db, current_user, run_id), failed=payload.failed)
    return timing.serialize(run)


@router.patch("/runs/{run_id}")
def update(run_id: int, payload: ResultsUpdate, current_user: User = Depends(admin_or_manager),
           db: Session = Depends(get_db)) -> dict:
    run = _guard(timing.update, db, _own(db, current_user, run_id), payload.model_dump(exclude_unset=True))
    return timing.serialize(run)


@router.delete("/runs/{run_id}")
def delete(run_id: int, current_user: User = Depends(admin_or_manager), db: Session = Depends(get_db)) -> dict:
    db.delete(_own(db, current_user, run_id))
    db.commit()
    return {"status": "deleted"}


@router.get("/export")
def export(current_user: User = Depends(admin_or_manager), db: Session = Depends(get_db)) -> Response:
    runs = list(reversed(_runs(db, current_user)))  # в CSV — по порядку проведения
    return Response(
        content=timing.export_csv(runs),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="timing.csv"'},
    )
