"""Аналитика пробелов в знаниях для владельцев базы (admin и manager).

Отчёт строится по журналу вопросов и оценкам «полезно / не полезно», см. services/gaps.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.auth import get_db, require_role
from app.db import ROLE_ADMIN, ROLE_MANAGER, User
from app.services import cad_demand, gaps
from app.state import embedder, kb_index, kb_lock


router = APIRouter(prefix="/api/admin/knowledge-gaps")

admin_or_manager = require_role(ROLE_ADMIN, ROLE_MANAGER)

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _report(db: Session, days: int) -> dict:
    with kb_lock:
        known = {path: doc.file_name for path, doc in kb_index.documents.items() if doc.searchable}
    report = gaps.build_report(db, days=days, embedder=embedder, known_documents=known)
    report["cad"] = cad_demand.build(db, days=days, embedder=embedder)
    return report


@router.get("")
def knowledge_gaps(
    days: int = Query(default=30, ge=1, le=365),
    current_user: User = Depends(admin_or_manager),
    db: Session = Depends(get_db),
) -> dict:
    return _report(db, days)


@router.get("/export")
def knowledge_gaps_export(
    days: int = Query(default=30, ge=1, le=365),
    current_user: User = Depends(admin_or_manager),
    db: Session = Depends(get_db),
) -> Response:
    content = gaps.build_report_xlsx(_report(db, days))
    return Response(
        content=content,
        media_type=XLSX_MIME,
        headers={"Content-Disposition": 'attachment; filename="knowledge_gaps.xlsx"'},
    )
