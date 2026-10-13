"""Точка входа FastAPI-приложения: создание app и регистрация роутеров.

Вся бизнес-логика разнесена по модулям:
- app/services/   — обработка вопросов
- app/routers/    — HTTP-эндпоинты
- app/state.py    — общие синглтоны (KB, LLM, dialog state, logger)
- app/schemas.py  — pydantic-модели запросов/ответов
- app/roles.py    — определение ролей участников в KB
"""

from __future__ import annotations

from fastapi import FastAPI
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
    profile,
    system,
    users_admin,
)
from app.services import cad
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

# Веб-редактор SolidCraft для вкладки «Конструктор» (ставится scripts/fetch_solidcraft.sh).
_editor = cad.editor_dir(settings.solidcraft_dir)
if _editor.is_dir():
    app.mount("/cad-editor", StaticFiles(directory=_editor, html=True), name="cad-editor")

# Раздача JS/CSS-бандла нового React-фронта (frontend/dist/assets) — если фронт собран.
system.attach_static_assets(app)


@app.on_event("startup")
def startup_event() -> None:
    settings.kb_root.mkdir(parents=True, exist_ok=True)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    with kb_lock:
        kb_index.build()
