"""Системные эндпоинты: корневая страница и health."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.state import kb_index, kb_lock, settings


router = APIRouter()


@router.get("/")
def root() -> FileResponse:
    index_path = settings.base_dir / "static" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(index_path)


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
