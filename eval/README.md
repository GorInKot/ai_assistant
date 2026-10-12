# Eval (локальная автопроверка качества)

Цель: быстро и повторяемо проверять качество ассистента после изменений.

## Что проверяется
- доля успешных кейсов (`pass_rate`)
- доля fallback-ответов (`fallback_rate`)
- соответствие источников ожидаемому процессу/документам
- корректная обработка неоднозначных вопросов и уточнений

Примечание: для кейсов, где важна именно генерация LLM/fallback-семантика, можно указать `skip_when_stub_llm: true`, чтобы не учитывать их в `--stub-llm` прогоне.

## Структура
- `eval/cases.json` — retrieval/QA-кейсы (источники, fallback, уточнения)
- `eval/intent_cases.json` — устаревшие кейсы intent-классификатора (раннер их пропускает, см. ниже)
- `scripts/run_eval.py` — скрипт прогона
- `eval/reports/` — отчеты прогонов (`latest.json`, `intent-latest.json`, baseline и т.д.)

## Авторизация и изоляция

`/api/ask` требует авторизации, поэтому раннер сам получает Bearer-токен:
- **local** — поднимает изолированную временную SQLite-БД (eval не читает и не
  мутирует dev `app_data.db`), строит KB in-process и минтит токен техническому
  пользователю. KB больше НЕ переиндексируется через `/api/reindex` (он под admin).
- **http** — регистрирует/логинит технического пользователя на запущенном сервере.

## Кейсы intent-классификатора (устарели)

Классификатор интентов удалён вместе с оформлением заявок (они в Портале). Кейсы
`kind: "intent"` в `eval/intent_cases.json` раннер пропускает со статусом skipped.
Файл оставлен только как held-out для генератора датасета `scripts/build_intent_dataset.py`.

## Запуск

### 1) Без сети/OpenAI (рекомендуется для регрессии retrieval)
```bash
python scripts/run_eval.py --mode local --stub-llm --output eval/reports/latest.json --fail-on-errors
```

### 2) Через запущенный локальный сервер
```bash
python scripts/run_eval.py --mode http --base-url http://127.0.0.1:8000 --output eval/reports/latest.json --fail-on-errors
```

## Baseline
Первый стабильный прогон сохраняйте отдельным файлом:
```bash
python scripts/run_eval.py --mode local --stub-llm --output eval/reports/baseline-local.json
```

Дальше сравнивайте `latest.json` с baseline по полям:
- `summary.pass_rate`
- `summary.fallback_rate`
- `results[*].ok` / `results[*].reasons`
