"""Целостность датасета для decision-модели интентов (Трек 2).

Главное, что здесь защищаем: тест честный — ни один шаблон не пересекается между
train/val/test, а реальные eval-кейсы остаются нетронутым held-out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_intent_dataset as ds  # noqa: E402


@pytest.fixture(scope="module")
def splits():
    return ds.build()


def test_build_is_deterministic():
    assert ds.build() == ds.build()


def test_labels_are_valid(splits):
    for rows in splits.values():
        for r in rows:
            assert r["intent"] in ds.INTENTS
            if r["intent"] == "create_request":
                assert r["request_type"] in ds.REQUEST_TYPES
            else:
                assert r["request_type"] == "none"


def test_templates_do_not_leak_between_splits(splits):
    owner: dict[str, str] = {}
    for name, rows in splits.items():
        for r in rows:
            key = (r["intent"], r["request_type"], r["template"])
            assert owner.setdefault(key, name) == name, f"шаблон в двух сплитах: {key}"


def test_texts_are_unique_across_dataset(splits):
    texts = [ds._norm(r["text"]) for rows in splits.values() for r in rows]
    assert len(texts) == len(set(texts))


def test_real_eval_cases_are_not_in_dataset(splits):
    eval_texts = {
        ds._norm(c["question"])
        for c in json.loads(ds.EVAL_CASES.read_text(encoding="utf-8"))
        if "question" in c
    }
    dataset_texts = {ds._norm(r["text"]) for rows in splits.values() for r in rows}
    assert not (eval_texts & dataset_texts)


def test_every_class_present_in_every_split(splits):
    for name, rows in splits.items():
        present = {r["intent"] for r in rows}
        assert present == set(ds.INTENTS), f"{name}: нет классов {set(ds.INTENTS) - present}"
    # каждый тип заявки должен быть и в train, и в test
    for name in ("train", "test"):
        types = {r["request_type"] for r in splits[name] if r["intent"] == "create_request"}
        assert types == set(ds.REQUEST_TYPES), name


def test_request_types_match_seeded_catalog():
    yaml_text = (ROOT / "app" / "data" / "request_types.yaml").read_text(encoding="utf-8")
    for slug in ds.REQUEST_TYPES:
        assert slug in yaml_text, f"{slug} отсутствует в каталоге заявок"
