"""Системные эндпоинты: корневая страница и health."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.state import kb_index, kb_lock, settings


router = APIRouter()


def _frontend_dist_path():
    """Путь к собранному React-фронту (frontend/dist).

    Если фронт собран — отдаём его. Если нет — fallback на старый static/index.html.
    Это позволяет работать сразу после клонирования репо: достаточно
    либо собрать фронт (npm run build), либо запустить как есть со старым UI.
    """
    return settings.base_dir / "frontend" / "dist"


@router.get("/")
def root() -> FileResponse:
    dist_index = _frontend_dist_path() / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index)

    legacy_index = settings.base_dir / "static" / "index.html"
    if legacy_index.exists():
        return FileResponse(legacy_index)

    raise HTTPException(
        status_code=404,
        detail="UI не найден. Соберите фронт: cd frontend && npm install && npm run build",
    )


@router.get("/api/health")
def health() -> dict[str, str | int]:
    with kb_lock:
        chunk_count = len(kb_index.chunks)
        document_count = len(kb_index.documents)

    return {
        "status": "ok",
        "documents": document_count,
        "chunks": chunk_count,
    }


def attach_static_assets(app) -> None:
    """Монтирует /assets из frontend/dist для раздачи Vite-бандла (JS/CSS).

    Вызывается из main.py после создания app. Если фронт не собран —
    маунт пропускается (старый static/index.html ассеты не использует).
    """
    dist = _frontend_dist_path()
    assets_dir = dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")
