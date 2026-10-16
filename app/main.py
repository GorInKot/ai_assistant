"""Точка входа FastAPI-приложения: создание app и регистрация роутеров.

Вся бизнес-логика разнесена по модулям:
- app/services/   — обработка вопросов
- app/routers/    — HTTP-эндпоинты
- app/state.py    — общие синглтоны (KB, LLM, dialog state, logger)
- app/schemas.py  — pydantic-модели запросов/ответов
- app/roles.py    — определение ролей участников в KB
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.routers import (
    actions,
    auth,
    cad as cad_router,
    chat,
    conversations,
    documents,
    knowledge_gaps,
    portal_sso,
    profile,
    system,
    users_admin,
)
from app.services import cad, route_drawing
from app.state import kb_index, kb_lock, settings


init_db()

app = FastAPI(title="Corporate AI Assistant MVP")

app.include_router(system.router)
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(documents.router)
app.include_router(knowledge_gaps.router)
app.include_router(cad_router.router)
app.include_router(actions.router)
app.include_router(profile.router)
app.include_router(users_admin.router)
app.include_router(portal_sso.router)


@app.middleware("http")
async def frame_policy(request: Request, call_next):
    # Встраивать ассистент в iframe можно только ему самому и порталу (PORTAL_ORIGINS).
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", f"frame-ancestors {portal_sso.frame_ancestors()}")
    return response

# Веб-редактор SolidCraft для вкладки «Конструктор» (ставится scripts/fetch_solidcraft.sh).
_editor = cad.editor_dir(settings.solidcraft_dir)
if _editor.is_dir():
    app.mount("/cad-editor", StaticFiles(directory=_editor, html=True), name="cad-editor")

# Веб-редактор CADCraft для вкладки «Чертёж трассы» (ставится scripts/fetch_cadcraft.sh).
_drawing_editor = route_drawing.editor_dir(settings.cadcraft_dir)
if _drawing_editor.is_dir():
    app.mount("/cadcraft-editor", StaticFiles(directory=_drawing_editor, html=True), name="cadcraft-editor")

# Раздача JS/CSS-бандла нового React-фронта (frontend/dist/assets) — если фронт собран.
system.attach_static_assets(app)


@app.on_event("startup")
def startup_event() -> None:
    settings.kb_root.mkdir(parents=True, exist_ok=True)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    with kb_lock:
        kb_index.build()
