"""Тесты локального эмбеддера и привязки дискового кэша KB к модели."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import pytest

from app.embeddings import OpenAICompatEmbedder, OpenAICompatEmbedderConfig
from app.kb import ChunkRecord, KnowledgeBaseIndex


def _embedder(handler, **cfg) -> OpenAICompatEmbedder:
    config = OpenAICompatEmbedderConfig(
        base_url="http://emb.local/v1", model="m", initial_backoff_sec=0.0, **cfg
    )
    return OpenAICompatEmbedder(config, transport=httpx.MockTransport(handler))


def _echo_handler(requests: list[dict]):
    """Вектор = [длина текста, номер в батче]; индексы отдаём вперемешку."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        data = [
            {"index": i, "embedding": [float(len(text)), float(i)]}
            for i, text in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    return handler


def test_documents_are_batched_and_keep_order():
    calls: list[dict] = []
    emb = _embedder(_echo_handler(calls), batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    matrix = emb.embed_documents(texts)
    assert [len(c["input"]) for c in calls] == [2, 2, 1]
    assert matrix[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]  # порядок сохранён


def test_prefixes_applied_to_query_and_documents():
    calls: list[dict] = []
    emb = _embedder(_echo_handler(calls), query_prefix="Q: ", doc_prefix="D: ")
    emb.embed_query("вопрос")
    emb.embed_documents(["текст"])
    assert calls[0]["input"] == ["Q: вопрос"]
    assert calls[1]["input"] == ["D: текст"]


def test_empty_document_gets_zero_vector():
    emb = _embedder(_echo_handler([]))
    matrix = emb.embed_documents(["текст", "   "])
    assert matrix[1].tolist() == [0.0, 0.0]


def test_failure_raises_after_retries():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503)

    emb = _embedder(handler, max_retries=3)
    with pytest.raises(RuntimeError, match="after 3 retries"):
        emb.embed_query("x")
    assert len(attempts) == 3


def test_model_id_depends_on_model_and_prefixes():
    a = _embedder(_echo_handler([])).model_id
    b = _embedder(_echo_handler([]), query_prefix="Q: ").model_id
    assert a != b


# ---------------- кэш KB привязан к модели ----------------


class _FakeEmbedder:
    def __init__(self, model_id: str, dim: int) -> None:
        self.model_id = model_id
        self.dim = dim

    def embed_documents(self, texts):
        return np.ones((len(texts), self.dim), dtype=np.float32)


def _chunk(text: str, n: int) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"c{n}", relative_path="d.md", text=text, page=None, section=None,
        process="p", token_counts=Counter(), token_total=1, section_tokens=set(),
        path_tokens=set(), role_mentions=0,
    )


def test_cache_key_depends_on_model():
    assert KnowledgeBaseIndex._chunk_hash("t", "a") != KnowledgeBaseIndex._chunk_hash("t", "b")


def test_switching_model_drops_old_vectors_and_keeps_cache_savable(tmp_path: Path):
    cache = tmp_path / "emb.npz"
    chunks = [_chunk("первый", 1), _chunk("второй", 2)]

    kb = KnowledgeBaseIndex(tmp_path, embedder=_FakeEmbedder("old", 256), embeddings_cache_path=cache)
    kb.chunks = chunks
    kb._rebuild_embeddings()
    assert kb.embedding_matrix.shape == (2, 256)

    # смена модели на другую размерность: старый кэш не должен подмешиваться
    kb2 = KnowledgeBaseIndex(tmp_path, embedder=_FakeEmbedder("new", 1024), embeddings_cache_path=cache)
    kb2.chunks = chunks
    kb2._rebuild_embeddings()
    assert kb2.embedding_matrix.shape == (2, 1024)

    saved = np.load(cache, allow_pickle=False)
    assert saved["vectors"].shape == (2, 1024)  # старые 256-мерные векторы вычищены


# ---------------- reasoning_effort для локальных LLM ----------------


class _CapturingCompletions:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = type("M", (), {"content": "ответ"})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


def _llm_with_fake_client(reasoning_effort):
    from app.llm import LLMService

    llm = LLMService("k", "m", base_url="http://x/v1", reasoning_effort=reasoning_effort)
    completions = _CapturingCompletions()
    llm.client = type("Cl", (), {"chat": type("Ch", (), {"completions": completions})()})()
    return llm, completions


def test_reasoning_effort_sent_only_when_configured():
    llm, completions = _llm_with_fake_client("none")
    llm._create(model="m", messages=[])
    assert completions.kwargs["extra_body"] == {"reasoning_effort": "none"}

    llm, completions = _llm_with_fake_client(None)
    llm._create(model="m", messages=[])
    assert "extra_body" not in completions.kwargs
