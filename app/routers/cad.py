"""Конструктор деталей: описание словами → 3D-модель (SolidCraft), см. services/cad.py."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import get_current_user
from app.services import cad
from app.state import llm_service, settings


router = APIRouter(prefix="/api/cad", dependencies=[Depends(get_current_user)])


class CadRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=cad.MAX_DESCRIPTION)


@router.get("/status")
def cad_status() -> dict:
    return {
        "available": cad.is_available(settings.solidcraft_dir),
        "version": cad.version(settings.solidcraft_dir),
        "editor": cad.editor_dir(settings.solidcraft_dir).is_dir(),
    }


@router.post("/generate")
def cad_generate(payload: CadRequest) -> dict:
    try:
        return cad.generate(payload.description, llm_service, settings.solidcraft_dir)
    except cad.CadUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except cad.CadSpecError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except cad.CadError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RuntimeError as exc:  # модель недоступна
        raise HTTPException(status_code=503, detail=str(exc)) from exc
