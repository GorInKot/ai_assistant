from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    kb_root: Path
    log_file: Path
    llm_api_key: str | None
    llm_model: str
    llm_base_url: str | None
    enable_llm_rerank: bool
    rerank_candidates: int
    rerank_top_n: int


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}



def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parent.parent
    kb_root_raw = os.getenv("KB_ROOT", "knowledge_base")
    log_file_raw = os.getenv("LOG_FILE", "logs/assistant.log")

    kb_root = (base_dir / kb_root_raw).resolve() if not Path(kb_root_raw).is_absolute() else Path(kb_root_raw)
    log_file = (base_dir / log_file_raw).resolve() if not Path(log_file_raw).is_absolute() else Path(log_file_raw)

    # LLM-бэкенд через OpenAI-совместимый API.
    # LLM_BASE_URL пуст -> OpenAI; задан -> любой совместимый сервер
    # (Groq: https://api.groq.com/openai/v1, Ollama: http://localhost:11434/v1).
    # Старые OPENAI_* поддерживаются как запасной вариант для обратной совместимости.
    llm_base_url = (os.getenv("LLM_BASE_URL") or "").strip() or None
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "llama-3.3-70b-versatile"

    return Settings(
        base_dir=base_dir,
        kb_root=kb_root,
        log_file=log_file,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        enable_llm_rerank=_as_bool(os.getenv("ENABLE_LLM_RERANK"), True),
        rerank_candidates=int(os.getenv("RERANK_CANDIDATES", "28")),
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "16")),
    )
