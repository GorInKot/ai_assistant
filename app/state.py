"""Глобальные сервисы и общее состояние процесса.

Здесь живут синглтоны, которыми пользуются роутеры:
- настройки приложения
- индекс KB + блокировка для атомарной переиндексации
- клиент LLM
- логгер запросов, хранилище действий, состояние диалогов
"""

from __future__ import annotations

from threading import Lock

from app.actions import ActionsStore
from app.config import load_settings
from app.dialog_state import DialogStateStore
from app.kb import KnowledgeBaseIndex
from app.llm import LLMService
from app.logging_utils import RequestLogger


settings = load_settings()
kb_index = KnowledgeBaseIndex(settings.kb_root)
kb_lock = Lock()
llm_service = LLMService(
    settings.llm_api_key,
    settings.llm_model,
    base_url=settings.llm_base_url,
    enable_rerank=settings.enable_llm_rerank,
    rerank_candidates=settings.rerank_candidates,
)
request_logger = RequestLogger(settings.log_file)
actions_store = ActionsStore(settings.log_file.parent / "actions.log")
dialog_state = DialogStateStore(ttl_minutes=20)
