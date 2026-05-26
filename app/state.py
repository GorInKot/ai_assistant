"""Глобальные сервисы и общее состояние процесса.

Здесь живут синглтоны, которыми пользуются роутеры:
- настройки приложения
- индекс KB + блокировка для атомарной переиндексации
- клиент LLM, клиент эмбеддингов (опционально)
- логгер запросов, хранилище действий, состояние диалогов
"""

from __future__ import annotations

import logging
from threading import Lock

from app.actions import ActionsStore
from app.config import load_settings
from app.dialog_state import DialogStateStore
from app.embeddings import YandexEmbedder, YandexEmbedderConfig
from app.kb import KnowledgeBaseIndex
from app.llm import LLMService
from app.logging_utils import RequestLogger


logger = logging.getLogger(__name__)

settings = load_settings()

# Embedder создаётся только если есть YC-ключи. Без них KB работает на BM25.
embedder: YandexEmbedder | None = None
if settings.yc_api_key and settings.yc_folder_id:
    embedder = YandexEmbedder(
        YandexEmbedderConfig(api_key=settings.yc_api_key, folder_id=settings.yc_folder_id)
    )
    logger.info("Yandex embedder enabled (folder_id=%s)", settings.yc_folder_id[:8] + "…")
else:
    logger.info("Yandex embedder disabled (YC_API_KEY/YC_FOLDER_ID not set)")

kb_index = KnowledgeBaseIndex(
    settings.kb_root,
    embedder=embedder,
    embeddings_cache_path=settings.embeddings_cache_path,
)
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
