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
    openai_api_key: str | None
    openai_model: str
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

    return Settings(
        base_dir=base_dir,
        kb_root=kb_root,
        log_file=log_file,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        enable_llm_rerank=_as_bool(os.getenv("ENABLE_LLM_RERANK"), True),
        rerank_candidates=int(os.getenv("RERANK_CANDIDATES", "28")),
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "16")),
    )
