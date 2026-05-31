"""Юнит-тесты эвристик распознавания вопросов про участников/роли.

Главная регресс-защита бага: паттерн `роль` без границы слова матчил
«конт-роль», из-за чего вопросы про *строительный контроль* (ключевой термин
продукта) ошибочно распознавались как «вопрос про участников» и получали список
ролей вместо ответа по сути. См. eval/cases.json::tsus_steps.

Эти тесты офлайн и детерминированы (regex), в отличие от eval, который требует
реального LLM и упирается в дневной лимит токенов.
"""

from __future__ import annotations

import pytest

from app.roles import is_participant_question


@pytest.mark.parametrize(
    "question",
    [
        "Как проходит строительный контроль в ЦУС?",
        "Сроки контроля качества",
        "контроль исполнения заявок",
        "Что такое строительный контроль?",
    ],
)
def test_control_question_not_treated_as_participant(question):
    # «контроль» НЕ должно триггерить ветку участников (баг 'контроль'~'роль').
    assert is_participant_question(question) is False


@pytest.mark.parametrize(
    "question",
    [
        "Кто участники процесса?",
        "Какие роли в процессе ЕКТП?",
        "Распределение ролей по этапам",
        "Кто заявитель и кто заказчик?",
        "Who are the participants of the process?",
    ],
)
def test_genuine_participant_questions_detected(question):
    assert is_participant_question(question) is True
