from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Query, Body, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.actions import ActionsStore
from app.config import load_settings
from app.dialog_state import DialogStateStore
from app.kb import KnowledgeBaseIndex, RetrievalResult
from app.llm import FALLBACK_EN, FALLBACK_RU, LLMService
from app.logging_utils import RequestLogger

from datetime import timedelta
from sqlalchemy.orm import Session
from app.db import init_db, User, SessionLocal
from app.auth import get_password_hash, verify_password, create_access_token, get_db, get_current_user

import os

init_db()

settings = load_settings()
kb_index = KnowledgeBaseIndex(settings.kb_root)
kb_lock = Lock()
llm_service = LLMService(
    settings.openai_api_key,
    settings.openai_model,
    enable_rerank=settings.enable_llm_rerank,
    rerank_candidates=settings.rerank_candidates,
)
request_logger = RequestLogger(settings.log_file)
actions_store = ActionsStore(settings.log_file.parent / "actions.log")
dialog_state = DialogStateStore(ttl_minutes=20)

print("FROM OS:", os.environ.get("OPENAI_API_KEY"))

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    no_exact_match: bool


class ActionCreateRequest(BaseModel):
    action_type: str = Field(..., min_length=2)
    process: str = Field(..., min_length=2)
    title: str = Field(..., min_length=2)
    details: str = Field(..., min_length=2)
    requester: str = Field(default="Не указан")


class DialogClearRequest(BaseModel):
    session_id: str | None = None


PARTICIPANT_QUERY_RE = re.compile(
    r"(участник\w*|актор\w*|роль\w*|заявител\w*|заказчик\w*|исполнител\w*|participant\w*|actor\w*|role\w*)",
    re.IGNORECASE,
)

ROLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Заявитель", re.compile(r"\bзаявител\w*\b", re.IGNORECASE)),
    ("Заказчик", re.compile(r"\bзаказчик\w*\b", re.IGNORECASE)),
    ("Диспетчер", re.compile(r"\bдиспетчер\w*\b", re.IGNORECASE)),
    ("Контроллер", re.compile(r"\bконтроллер\w*\b", re.IGNORECASE)),
    ("Специалист техподдержки", re.compile(r"специалист\s+техпод", re.IGNORECASE)),
    ("Редактор справочников", re.compile(r"редактор\s+справоч", re.IGNORECASE)),
    ("Администратор", re.compile(r"\bадминистратор\w*\b", re.IGNORECASE)),
]



def detect_language(question: str) -> str:
    en_count = len(re.findall(r"[A-Za-z]", question))
    ru_count = len(re.findall(r"[А-Яа-яЁё]", question))
    if en_count > max(ru_count * 1.2, 3):
        return "en"
    return "ru"


def is_participant_question(question: str) -> bool:
    return bool(PARTICIPANT_QUERY_RE.search(question))


def extract_participants_from_results(
    results: list[RetrievalResult],
    process_hint: str | None = None,
    max_items: int = 10,
) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()

    for result in results:
        chunk = result.chunk
        if process_hint and chunk.process != process_hint:
            continue

        for label, pattern in ROLE_PATTERNS:
            if label in seen:
                continue
            if pattern.search(chunk.text):
                seen.add(label)
                items.append(label)
                if len(items) >= max_items:
                    return items

    return items


def build_participants_answer(language: str, participants: list[str], process_hint: str | None) -> str:
    subject = process_hint.replace("_", " ") if process_hint else "указанном процессе"
    if language == "en":
        lines = "\n".join(f"- {item}" for item in participants)
        return f"Found process participant roles in {subject}:\n{lines}"

    lines = "\n".join(f"- {item}" for item in participants)
    return f"В базе знаний найдены участники (роли) заявочного процесса в {subject}:\n{lines}"


def normalize_session_id(session_id: str | None) -> str:
    value = (session_id or "").strip()
    if not value:
        return "local-default"
    if len(value) > 96:
        return value[:96]
    return value


def _source_extension(source: dict) -> str:
    return Path(str(source.get("relative_path", ""))).suffix.lower()


def _source_process(source: dict) -> str:
    relative_path = str(source.get("relative_path", ""))
    return relative_path.split("/", 1)[0] if "/" in relative_path else "Общее"


def _dominant_context_process(context_results: list[RetrievalResult]) -> str | None:
    if not context_results:
        return None

    counts: Counter[str] = Counter(item.chunk.process for item in context_results)
    top_process, top_count = counts.most_common(1)[0]
    total = sum(counts.values())
    share = top_count / max(total, 1)

    if len(counts) == 1:
        return top_process
    if top_count >= 2 and share >= 0.6:
        return top_process
    return None


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for source in sources:
        key = str(source.get("relative_path", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


def select_display_sources(
    question: str,
    process_hint: str | None,
    context_results: list[RetrievalResult],
    context_sources: list[dict],
    related_sources: list[dict],
) -> list[dict]:
    # In UI sources we show business documents, not service markdown stubs.
    candidates = [*context_sources, *related_sources]
    non_md_sources = [source for source in candidates if _source_extension(source) != ".md"]

    target_process = process_hint or _dominant_context_process(context_results)
    if target_process:
        process_only = [
            source
            for source in non_md_sources
            if _source_process(source) == target_process
        ]
        if process_only:
            non_md_sources = process_only

    non_md_sources = _dedupe_sources(non_md_sources)
    non_md_sources.sort(
        key=lambda source: (
            0 if source.get("source_type") == "context" else 1,
            source.get("relative_path", ""),
        )
    )

    if non_md_sources:
        return non_md_sources[:6]

    # Fallback: suggest available non-markdown docs from the detected process.
    process_for_fallback = target_process or process_hint
    if process_for_fallback:
        process_docs = kb_index.list_documents(query=question, process=process_for_fallback, forms_only=False)
        fallback_docs = [
            {
                "file_name": doc["file_name"],
                "relative_path": doc["relative_path"],
                "url": doc["url"],
                "download_url": doc["download_url"],
                "pages": [],
                "sections": [],
                "source_type": "related",
            }
            for doc in process_docs
            if Path(str(doc["relative_path"])).suffix.lower() != ".md"
        ]
        return _dedupe_sources(fallback_docs)[:4]

    return []



def build_fallback_answer(language: str, related_docs: list[dict], reason: str = "missing") -> str:
    ambiguous_suffix_en = " Please specify the process or system first."
    ambiguous_suffix_ru = " Уточните, пожалуйста, процесс или систему (например: ЕКТП, ЦУС, обучение/медосмотр)."

    if language == "en":
        prefix = FALLBACK_EN if reason == "missing" else "The request is ambiguous across multiple processes."
        if related_docs:
            return (
                f"{prefix}\n\n"
                "Please clarify the request (process, role, system, or document name), "
                "or review the related documents from the source list."
                + (ambiguous_suffix_en if reason == "ambiguous" else "")
            )
        return (
            f"{prefix}\n\n"
            "Please clarify the request (process, role, system, or document name)."
            + (ambiguous_suffix_en if reason == "ambiguous" else "")
        )

    prefix = FALLBACK_RU if reason == "missing" else "Запрос неоднозначен и может относиться к нескольким процессам."
    if related_docs:
        return (
            f"{prefix}\n\n"
            "Уточните вопрос (процесс, роль, систему или название документа) "
            "или откройте релевантные документы из блока источников."
            + (ambiguous_suffix_ru if reason == "ambiguous" else "")
        )

    return (
        f"{prefix}\n\n"
        "Уточните вопрос (процесс, роль, систему или название документа)."
        + (ambiguous_suffix_ru if reason == "ambiguous" else "")
    )


app = FastAPI(title="Corporate AI Assistant MVP")


@app.on_event("startup")
def startup_event() -> None:
    settings.kb_root.mkdir(parents=True, exist_ok=True)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    with kb_lock:
        kb_index.build()


@app.get("/")
def root() -> FileResponse:
    index_path = settings.base_dir / "static" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> dict[str, str | int]:
    with kb_lock:
        chunk_count = len(kb_index.chunks)
        document_count = len(kb_index.documents)

    return {
        "status": "ok",
        "documents": document_count,
        "chunks": chunk_count,
    }


@app.post("/api/reindex")
def reindex() -> dict[str, int | str]:
    with kb_lock:
        kb_index.build()
        documents = len(kb_index.documents)
        chunks = len(kb_index.chunks)

    return {"status": "reindexed", "documents": documents, "chunks": chunks}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    raw_question = payload.question.strip()
    if not raw_question:
        raise HTTPException(status_code=400, detail="Question is empty")

    session_id = normalize_session_id(payload.session_id)
    effective_question, _ = dialog_state.merge_with_pending(session_id, raw_question)
    language = detect_language(effective_question)

    with kb_lock:
        process_hint = kb_index.detect_process_hint(effective_question)
        lexical_results = kb_index.retrieve(
            effective_question,
            top_k=max(50, settings.rerank_candidates * 2),
        )
        reranked_results = llm_service.rerank_results(
            effective_question,
            lexical_results,
            top_n=settings.rerank_top_n,
        )

        intent = kb_index.classify_query_intent(effective_question)
        context_results = kb_index.select_context_results(effective_question, reranked_results, max_chunks=5)
        has_strong_context = kb_index.is_context_strong(context_results)
        process_ambiguous = kb_index.is_process_ambiguous(context_results)
        context_sources = kb_index.build_sources_from_results(context_results)

        exclude_paths = {item["relative_path"] for item in context_sources}
        related_sources = kb_index.find_related_documents(
            effective_question,
            limit=4,
            exclude_paths=exclude_paths,
        )
        display_sources = select_display_sources(
            question=effective_question,
            process_hint=process_hint,
            context_results=context_results,
            context_sources=context_sources,
            related_sources=related_sources,
        )

    final_sources = display_sources

    if not has_strong_context or (process_ambiguous and process_hint is None):
        reason = "ambiguous" if process_ambiguous and process_hint is None else "missing"
        answer = build_fallback_answer(language, final_sources, reason=reason)
        if reason == "ambiguous":
            dialog_state.set_pending(session_id, effective_question)
        request_logger.log(raw_question, final_sources, answer=answer)
        return AskResponse(answer=answer, sources=final_sources, no_exact_match=True)

    if is_participant_question(effective_question):
        participants = extract_participants_from_results(
            reranked_results[: min(len(reranked_results), 24)],
            process_hint=process_hint,
        )
        if len(participants) >= 2:
            answer = build_participants_answer(language, participants, process_hint=process_hint)
            dialog_state.clear(session_id)
            request_logger.log(raw_question, final_sources, answer=answer)
            return AskResponse(answer=answer, sources=final_sources, no_exact_match=False)

    try:
        answer = llm_service.generate_answer(
            effective_question, 
            context_results,
            intent=intent
            )
        
    except RuntimeError as exc:
        # 
        print("LLM Runtime error: ", repr(exc))
        raise HTTPException(
            status_code=503,
            detail=str(exc),   # ← больше НЕ пишем "API key is required"
        ) from exc
    except Exception as exc:  
        print("LLM Unexpected error:", repr(exc))
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    no_exact_match = FALLBACK_RU.lower() in answer.lower() or FALLBACK_EN.lower() in answer.lower()
    dialog_state.clear(session_id)
    request_logger.log(raw_question, final_sources, answer=answer)
    return AskResponse(answer=answer, sources=final_sources, no_exact_match=no_exact_match)


@app.post("/api/dialog/clear")
def clear_dialog_state(payload: DialogClearRequest) -> dict[str, str]:
    session_id = normalize_session_id(payload.session_id)
    dialog_state.clear(session_id)
    return {"status": "cleared"}


@app.get("/api/debug/retrieval")
def debug_retrieval(
    q: str = Query(..., min_length=2),
    limit: int = Query(default=12, ge=1, le=50),
) -> dict[str, object]:
    def serialize(results: list[RetrievalResult], size: int) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for result in results[:size]:
            chunk = result.chunk
            payload.append(
                {
                    "score": round(result.score, 4),
                    "coverage": round(result.coverage, 4),
                    "relative_path": chunk.relative_path,
                    "page": chunk.page,
                    "section": chunk.section,
                    "snippet": chunk.text[:260],
                }
            )
        return payload

    with kb_lock:
        lexical_results = kb_index.retrieve(q, top_k=max(40, settings.rerank_candidates))
        reranked_results = llm_service.rerank_results(q, lexical_results, top_n=max(limit, 10))
        context_results = kb_index.select_context_results(q, reranked_results, max_chunks=min(limit, 8))

    return {
        "query": q,
        "intent": kb_index.classify_query_intent(q),
        "lexical": serialize(lexical_results, limit),
        "reranked": serialize(reranked_results, limit),
        "context": serialize(context_results, min(limit, 8)),
    }


@app.get("/api/documents")
def list_documents(
    q: str | None = Query(default=None),
    process: str | None = Query(default=None),
    forms_only: int = Query(default=0),
) -> dict[str, list[dict]]:
    with kb_lock:
        items = kb_index.list_documents(query=q, process=process, forms_only=bool(forms_only))
    return {"documents": items}


@app.get("/api/actions")
def list_actions(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, list[dict]]:
    return {"actions": actions_store.list_actions(limit=limit)}


@app.post("/api/actions")
def create_action(payload: ActionCreateRequest) -> dict[str, object]:
    record = actions_store.create_action(
        action_type=payload.action_type.strip(),
        process=payload.process.strip(),
        title=payload.title.strip(),
        details=payload.details.strip(),
        requester=payload.requester.strip(),
    )
    return {
        "status": "created",
        "message": "Действие зарегистрировано локально (MVP, без интеграции с корпоративными ИС).",
        "action": {
            "action_id": record.action_id,
            "action_type": record.action_type,
            "process": record.process,
            "title": record.title,
            "details": record.details,
            "requester": record.requester,
            "status": record.status,
            "created_at": record.created_at,
        },
    }


@app.get("/api/files/{file_path:path}")
def get_file(file_path: str, download: int = 0) -> FileResponse:
    kb_root_resolved = settings.kb_root.resolve()
    requested_path = (kb_root_resolved / file_path).resolve()

    if kb_root_resolved not in requested_path.parents:
        raise HTTPException(status_code=403, detail="Path is outside knowledge base")

    if not requested_path.exists() or not requested_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if download:
        return FileResponse(
            path=requested_path,
            filename=requested_path.name,
            media_type="application/octet-stream",
        )

    return FileResponse(path=requested_path)

# New code
class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str | None = None

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@app.post("/api/auth/register", response_model=TokenResponse)
def register_user(payload: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Пользователь с таким email уже существует")
    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=get_password_hash(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)

@app.post("/api/auth/login", response_model=TokenResponse)
def login_user(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)

@app.get("/api/user/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "full_name": current_user.full_name,
        "created_at": current_user.created_at,
    }