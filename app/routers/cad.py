"""Конструктор: детали (3D, SolidCraft, services/cad.py) и чертёж трассы (2D, CADCraft, services/route_drawing.py).

Каждый запрос пишется в историю пользователя (services/cad_history.py): описание, исход и
схема. Картинки и файлы не хранятся, результат собирается заново.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_db
from app.db import User
from app.services import cad, cad_history, route_drawing
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
        "drawing": {
            "available": route_drawing.is_available(settings.cadcraft_dir),
            "version": route_drawing.version(settings.cadcraft_dir),
            "editor": route_drawing.editor_dir(settings.cadcraft_dir).is_dir(),
        },
    }


def _run(db: Session, user: User, description: str, kind: str, generate) -> dict:
    """Выполнить построение и записать запрос в историю (кроме недоступности модели или программы)."""
    try:
        result = generate()
    except (cad.CadUnavailable, route_drawing.CadcraftUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (cad.CadSpecError, route_drawing.RouteSpecError) as exc:
        cad_history.record(db, user.id, description, cad_history.OUTCOME_ERROR, kind=kind, detail=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (cad.CadError, route_drawing.CadcraftError) as exc:
        cad_history.record(db, user.id, description, cad_history.OUTCOME_ERROR, kind=kind, detail=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RuntimeError as exc:  # модель недоступна — это не запрос пользователя, в историю не пишем
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if result.get("unsupported"):
        cad_history.record(db, user.id, description, cad_history.OUTCOME_UNSUPPORTED, kind=kind,
                           detail=result["unsupported"])
    elif result.get("clarify"):
        cad_history.record(db, user.id, description, cad_history.OUTCOME_CLARIFY, kind=kind,
                           detail=result["clarify"])
    else:
        cad_history.record(db, user.id, description, cad_history.OUTCOME_BUILT, kind=kind,
                           title=result["title"], spec=result["spec"], notes=result["assumptions"])
    return result


@router.post("/generate")
def cad_generate(
    payload: CadGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return _run(db, current_user, payload.description, cad_history.KIND_PART,
                lambda: cad.generate(payload.description, llm_service, settings.solidcraft_dir))


@router.post("/route")
def route_generate(
    payload: CadGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return _run(db, current_user, payload.description, cad_history.KIND_ROUTE,
                lambda: route_drawing.generate(payload.description, llm_service, settings.cadcraft_dir))


@router.get("/history")
def cad_history_list(
    limit: int = Query(default=cad_history.DEFAULT_LIMIT, ge=1, le=cad_history.MAX_PER_USER),
    kind: str = Query(default=cad_history.KIND_PART),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if kind not in cad_history.KINDS:
        raise HTTPException(status_code=422, detail=f"kind: одно из {', '.join(cad_history.KINDS)}")
    return {"items": cad_history.list_for_user(db, current_user.id, limit, kind)}


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
    """Собрать результат из записи истории заново по сохранённой схеме — без модели."""
    row = cad_history.get_own(db, current_user.id, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    if row.outcome != cad_history.OUTCOME_BUILT or not row.spec_json:
        raise HTTPException(status_code=400, detail="По этому запросу ничего не строилось")
    spec, notes = cad_history.stored_spec(row)
    try:
        if row.kind == cad_history.KIND_ROUTE:
            return route_drawing.rebuild(spec, settings.cadcraft_dir, notes=notes)
        return cad.rebuild(spec, settings.solidcraft_dir, notes=notes)
    except (cad.CadUnavailable, route_drawing.CadcraftUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (cad.CadSpecError, route_drawing.RouteSpecError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (cad.CadError, route_drawing.CadcraftError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
