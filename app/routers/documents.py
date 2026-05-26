"""Работа с документами KB: список, файлы, переиндексация, отладка retrieval."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.kb import RetrievalResult
from app.state import kb_index, kb_lock, llm_service, settings


router = APIRouter(prefix="/api")


@router.post("/reindex")
def reindex() -> dict[str, int | str]:
    with kb_lock:
        kb_index.build()
        documents = len(kb_index.documents)
        chunks = len(kb_index.chunks)
    return {"status": "reindexed", "documents": documents, "chunks": chunks}


@router.get("/documents")
def list_documents(
    q: str | None = Query(default=None),
    process: str | None = Query(default=None),
    forms_only: int = Query(default=0),
) -> dict[str, list[dict]]:
    with kb_lock:
        items = kb_index.list_documents(query=q, process=process, forms_only=bool(forms_only))
    return {"documents": items}


@router.get("/debug/retrieval")
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


@router.get("/files/{file_path:path}")
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
