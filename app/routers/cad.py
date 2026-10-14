"""Конструктор деталей: описание словами → 3D-модель (SolidCraft), см. services/cad.py.

Каждый запрос пишется в историю пользователя (services/cad_history.py): описание, исход и
спецификация детали. Картинки и STEP не хранятся, деталь собирается заново.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_db
from app.db import User
from app.services import cad, cad_history
from app.state import llm_service, settings


router = APIRouter(prefix="/api/cad", dependencies=[Depends(get_current_user)])


class CadGenerateRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=cad.MAX_DESCRIPTION)


@router.get("/status")
def cad_status() -> dict:
    return {
        "available": cad.is_available(settings.solidcraft_dir),
        "version": cad.version(settings.solidcraft_dir),
        "editor": cad.editor_dir(settings.solidcraft_dir).is_dir(),
    }


@router.post("/generate")
def cad_generate(
    payload: CadGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    description = payload.description
    try:
        result = cad.generate(description, llm_service, settings.solidcraft_dir)
    except cad.CadUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except cad.CadSpecError as exc:
        cad_history.record(db, current_user.id, description, cad_history.OUTCOME_ERROR, detail=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except cad.CadError as exc:
        cad_history.record(db, current_user.id, description, cad_history.OUTCOME_ERROR, detail=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RuntimeError as exc:  # модель недоступна — это не запрос пользователя, в историю не пишем
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if result.get("unsupported"):
        cad_history.record(db, current_user.id, description, cad_history.OUTCOME_UNSUPPORTED,
                           detail=result["unsupported"])
    elif result.get("clarify"):
        cad_history.record(db, current_user.id, description, cad_history.OUTCOME_CLARIFY, detail=result["clarify"])
    else:
        cad_history.record(db, current_user.id, description, cad_history.OUTCOME_BUILT,
                           title=result["title"], spec=result["spec"], notes=result["assumptions"])
    return result


@router.get("/history")
def cad_history_list(
    limit: int = Query(default=cad_history.DEFAULT_LIMIT, ge=1, le=cad_history.MAX_PER_USER),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return {"items": cad_history.list_for_user(db, current_user.id, limit)}


@router.delete("/history/{request_id}")
def cad_history_delete(
    request_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    row = cad_history.get_own(db, current_user.id, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    db.delete(row)
    db.commit()
    return {"status": "deleted"}


@router.post("/history/{request_id}/rebuild")
def cad_history_rebuild(
    request_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Собрать деталь из записи истории заново по сохранённой спецификации — без модели."""
    row = cad_history.get_own(db, current_user.id, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    if row.outcome != cad_history.OUTCOME_BUILT or not row.spec_json:
        raise HTTPException(status_code=400, detail="По этому запросу деталь не строилась")
    spec, notes = cad_history.stored_spec(row)
    try:
        return cad.rebuild(spec, settings.solidcraft_dir, notes=notes)
    except cad.CadUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except cad.CadSpecError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except cad.CadError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
