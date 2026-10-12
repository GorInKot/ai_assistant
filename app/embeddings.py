"""Клиенты для векторных эмбеддингов: локальный (OpenAI-совместимый) и Yandex.

Локальный: `OpenAICompatEmbedder` — любой сервер с эндпоинтом `/v1/embeddings`
(Ollama, LM Studio, vLLM). Данные не покидают периметр.
Yandex: `YandexEmbedder` — облачный вариант, оставлен как опция.

Yandex API: https://yandex.cloud/ru/docs/foundation-models/embeddings/api-ref/Embeddings/textEmbedding
Модели:
  - text-search-doc — для индексации документов
  - text-search-query — для запросов пользователя
Размерность: 256.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import httpx
import numpy as np


logger = logging.getLogger(__name__)


YANDEX_EMBEDDINGS_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"


@dataclass
class YandexEmbedderConfig:
    api_key: str
    folder_id: str
    doc_model: str = "text-search-doc"
    query_model: str = "text-search-query"
    model_version: str = "latest"
    timeout_sec: float = 30.0
    max_workers: int = 4
    max_retries: int = 4
    initial_backoff_sec: float = 0.5
    max_text_chars: int = 2000


class YandexEmbedder:
    """Синхронный клиент с пулом воркеров и retry на 429/5xx.

    Yandex API принимает один текст за запрос — батчинг делаем сами через ThreadPoolExecutor.
    Лимиты на тарифе бесплатном — порядка единиц RPS, поэтому max_workers по умолчанию 4.
    """

    def __init__(self, config: YandexEmbedderConfig) -> None:
        self.config = config
        self._client = httpx.Client(
            timeout=config.timeout_sec,
            headers={
                "Authorization": f"Api-Key {config.api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def model_id(self) -> str:
        return f"yandex:{self.config.doc_model}/{self.config.query_model}/{self.config.model_version}"

    def close(self) -> None:
        self._client.close()

    def _model_uri(self, model_name: str) -> str:
        return f"emb://{self.config.folder_id}/{model_name}/{self.config.model_version}"

    def _embed_single(self, text: str, model_name: str) -> np.ndarray:
        # Yandex отказывает на пустых строках. Чистим и обрезаем до лимита (длинные тексты
        # вряд ли дадут лучший эмбеддинг и едят тариф).
        payload_text = text.strip()
        if not payload_text:
            return np.zeros(256, dtype=np.float32)
        if len(payload_text) > self.config.max_text_chars:
            payload_text = payload_text[: self.config.max_text_chars]

        body = {"modelUri": self._model_uri(model_name), "text": payload_text}

        backoff = self.config.initial_backoff_sec
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.post(YANDEX_EMBEDDINGS_URL, json=body)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    logger.warning(
                        "Yandex embeddings transient %s on attempt %d, backing off %.1fs",
                        response.status_code,
                        attempt + 1,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding") or []
                if not embedding:
                    raise ValueError(f"Empty embedding in response: {data!r}")
                return np.asarray(embedding, dtype=np.float32)
            except httpx.HTTPError as err:
                last_err = err
                logger.warning("Yandex embeddings HTTP error on attempt %d: %s", attempt + 1, err)
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(
            f"Yandex embeddings failed after {self.config.max_retries} retries: {last_err!r}"
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_single(text, self.config.query_model)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 256), dtype=np.float32)

        results: list[tuple[int, np.ndarray]] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            future_to_index = {
                pool.submit(self._embed_single, text, self.config.doc_model): idx
                for idx, text in enumerate(texts)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    vector = future.result()
                except Exception as err:
                    logger.error("Embedding failed for chunk index %d: %s", idx, err)
                    vector = np.zeros(256, dtype=np.float32)
                results.append((idx, vector))

        results.sort(key=lambda item: item[0])
        matrix = np.stack([vec for _, vec in results], axis=0)
        return matrix


@dataclass
class OpenAICompatEmbedderConfig:
    base_url: str  # например http://127.0.0.1:11434/v1 (Ollama)
    model: str  # например qwen3-embedding:0.6b или bge-m3
    api_key: str | None = None  # локальным серверам обычно не нужен
    # Модели с инструкциями (Qwen3-Embedding, E5) ждут префикс у запроса/документа.
    query_prefix: str = ""
    doc_prefix: str = ""
    timeout_sec: float = 120.0  # первый запрос грузит модель в память — может быть долгим
    batch_size: int = 16
    max_retries: int = 3
    initial_backoff_sec: float = 0.5
    max_text_chars: int = 2000


class OpenAICompatEmbedder:
    """Эмбеддинги через OpenAI-совместимый `/embeddings` (батчами).

    Совместим по интерфейсу с `YandexEmbedder` (embed_query / embed_documents),
    поэтому индекс KB от выбора провайдера не зависит. `model_id` попадает в ключ
    дискового кэша: при смене модели старые векторы не подмешиваются к новым.
    """

    def __init__(self, config: OpenAICompatEmbedderConfig, *, transport: httpx.BaseTransport | None = None) -> None:
        self.config = config
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.Client(timeout=config.timeout_sec, headers=headers, transport=transport)
        self._url = config.base_url.rstrip("/") + "/embeddings"

    @property
    def model_id(self) -> str:
        return f"openai-compat:{self.config.model}|q={self.config.query_prefix}|d={self.config.doc_prefix}"

    def close(self) -> None:
        self._client.close()

    def _prepare(self, text: str, prefix: str) -> str:
        cleaned = text.strip()[: self.config.max_text_chars]
        # Пустую строку сервер может отвергнуть — заменяем пробелом, вектор отбросим ниже.
        return (prefix + cleaned) if cleaned else " "

    def _embed_batch(self, inputs: list[str]) -> list[np.ndarray]:
        body = {"model": self.config.model, "input": inputs}
        backoff = self.config.initial_backoff_sec
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.post(self._url, json=body)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    last_err = RuntimeError(f"HTTP {response.status_code}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                response.raise_for_status()
                items = sorted(response.json()["data"], key=lambda it: it["index"])
                vectors = [np.asarray(it["embedding"], dtype=np.float32) for it in items]
                if len(vectors) != len(inputs):
                    raise ValueError(f"Ожидали {len(inputs)} векторов, получили {len(vectors)}")
                return vectors
            except (httpx.HTTPError, KeyError, ValueError) as err:
                last_err = err
                logger.warning("Embeddings error on attempt %d: %s", attempt + 1, err)
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError(f"Embeddings failed after {self.config.max_retries} retries: {last_err!r}")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_batch([self._prepare(text, self.config.query_prefix)])[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors: list[np.ndarray] = []
        size = self.config.batch_size
        for start in range(0, len(texts), size):
            chunk = texts[start : start + size]
            vectors.extend(self._embed_batch([self._prepare(t, self.config.doc_prefix) for t in chunk]))
        # Пустые тексты — нулевой вектор (в кэш/поиск такие строки не должны попадать).
        dim = vectors[0].shape[0]
        for i, text in enumerate(texts):
            if not text.strip():
                vectors[i] = np.zeros(dim, dtype=np.float32)
        return np.stack(vectors, axis=0)


def cosine_similarity_topk(query_vec: np.ndarray, doc_matrix: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Возвращает top_k индексов с косинусной близостью.

    Считает обе нормы каждый раз — для нашего размера индекса (тысячи векторов)
    это не узкое место. Если индекс вырастет — стоит закэшировать норму doc_matrix.
    """
    if doc_matrix.shape[0] == 0 or query_vec.size == 0:
        return []

    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return []

    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    safe_norms = np.where(doc_norms == 0, 1.0, doc_norms)
    similarities = (doc_matrix @ query_vec) / (safe_norms * q_norm)

    # Обнуляем строки, у которых нулевая норма (битые/пустые эмбеддинги).
    similarities = np.where(doc_norms == 0, -1.0, similarities)

    if top_k >= len(similarities):
        order = np.argsort(-similarities)
    else:
        # argpartition быстрее argsort при больших массивах.
        partial = np.argpartition(-similarities, top_k)[:top_k]
        order = partial[np.argsort(-similarities[partial])]

    return [(int(idx), float(similarities[idx])) for idx in order]
