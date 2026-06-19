# Текущий этап работы

> Снимок состояния на 2026-06-19. Используется как продолжение между сессиями.

## Этап 6 — работа с документами в чате (✅ закрыт)

См. [RoadMap.md](RoadMap.md). Реализовано **загрузкой файлов прямо в чат** (вложение к сообщению), а не отдельными REST-ручками — по запросу пользователя «функционал, как у любого ИИ-ассистента». Ассистент по сообщению + типу/числу файлов сам выбирает операцию.

| # | Возможность | Статус |
|---|---|---|
| 6.A.1 | Excel — извлечение данных (листы, заголовки, строки; пропуск титульных строк) | ✅ |
| 6.A.2 | Excel — объединение с проверкой идентичности структуры → `merged.xlsx` | ✅ |
| 6.A.3 | Excel — поиск различий (строки/ячейки/структура) | ✅ |
| 6.B.1 | Word — извлечение (текст, заголовки, таблицы) | ✅ |
| 6.B.2 | Word — краткая выжимка через LLM (`summarize_text`) | ✅ |
| 6.B.3 | Word — сравнение файлов (diff по абзацам) | ✅ |

**Ключевые решения:**
- Файлы обрабатываются **в памяти и НЕ сохраняются на сервере** (осознанный принцип для корпоративных данных).
- Вывод различий — текстовый отчёт в чате; объединение — скачиваемый `merged.xlsx` (base64).
- Excel-структура — сверка по именам колонок; Word-diff — по абзацам, только текст.
- `.doc` (старый бинарный) — вне скоупа, поддержаны `.xlsx`/`.docx`.
- Превью таблиц рендерятся как fenced-блоки (фронт без remark-gfm).

**Новые/изменённые файлы:**
- `app/services/document_processing.py` — ядро разбора Excel/Word (parse_xlsx, merge_workbooks, build_xlsx, diff_workbooks, parse_docx, diff_docx).
- `app/services/chat_documents.py` — маршрутизация сообщение+файлы → операция, markdown-ответ.
- `app/routers/chat.py` — `POST /api/ask-files` (multipart, сохраняет сообщения в беседе).
- `app/llm.py` — `summarize_text` (try/except → RuntimeError → 503 при недоступном LLM).
- `app/schemas.py` — `AskAttachment` + `attachment` в `AskResponse`.
- Фронт: `Composer.tsx` (📎 пикер), `ChatPage.tsx`, `MessageList.tsx` (download), `client.ts` (`uploadMany` + фикс Content-Type для FormData), `types.ts`.
- `tests/test_chat_documents.py` — 12 тестов (роутинг + интеграция `/api/ask-files`).

**Качество:** pytest **39 passed**.

**Дальше:** Этап 7 (автозаполнение шаблонов заявок) — ждёт входных данных от пользователя (формат шаблонов, механика маппинга, формат вывода).

## Последние коммиты

```
b2950d7  Email-уведомления ответственному о новых заявках
032975f  Security: eval http-режим — случайные креды + guard на не-локальный хост
38ffe3f  Alembic для миграций БД вместо create_all + ручных ALTER
fcd9917  Опциональные слоты заявок с возможностью «пропустить»
075352d  Тесты + eval: smoke-тесты, intent-eval и фиксы retrieval
0239073  Postgres вместо SQLite на проде
e2a5fda  Деплой на Render через Docker (multi-stage: node→python)
4c4d16f  Admin-UI для каталога типов заявок (БД вместо YAML)
bc06384  KB: расширение законодательством РФ — 13 новых документов
3f77fbf  Security: 3 фикса по итогам /security-review
```

Все коммиты запушены в `origin/main`. **Роадмап Этапа 5 (см. [RoadMap05.md](RoadMap05.md)) полностью закрыт.**

## Статус роадмапа

| Пункт | Статус |
|---|---|
| Pytest smoke-тесты | ✅ |
| Eval cases (intent classifier + slot-filling) | ✅ |
| Опциональные слоты с «пропустить» | ✅ |
| Alembic для миграций БД | ✅ |
| Email-уведомления о заявках | ✅ |

**Качество:** pytest **27 passed**; retrieval eval **12/12**; intent+slot eval **19/19**; /security-review чист.

## Сделано в последней сессии (2026-05-31)

### Тесты + eval + фиксы retrieval (`075352d`)
- `tests/` — pytest smoke (полный путь заявки через чат, анонимка, auth 401/403) + юнит-тесты `is_participant_question`. Изоляция на временной БД, LLM замокан. `pytest.ini`, `requirements-dev.txt`.
- `scripts/run_eval.py` — починен после security-фикса: авторизация к `/api/ask`, изоляция в temp-БД, KB in-process; новый `kind: "intent"` (прямой вызов `classify_intent`). `eval/intent_cases.json`.
- 2 реальных бага исправлены:
  - **`роль` без границы слова** ([roles.py](app/roles.py)): «конт**роль**» матчил паттерн → вопросы про стройконтроль отвечали списком ролей. Фикс `\bрол[ьиея]\w*`.
  - **уточнение перехватывалось intent-классификатором** ([ask_service.py](app/services/ask_service.py)): «по ЕКТП» запускало заявку. Фикс: `merge_with_pending` до классификации; вопрос про участников без процесса → просим уточнить.
- Ожидания `eval/cases.json` обновлены под текущий KB (переименованные документы, новый список ролей ЕКТП).

### Опциональные слоты с «пропустить» (`fcd9917`)
- Спрашиваются все слоты по порядку; у необязательных подсказка о пропуске. Skip-токены (`пропустить`, `-`, …) пропускают optional; обязательный — переспрос.
- ⚠️ Нюанс (поймал только реальный LLM): «пропустить» классификатор принимает за `cancel` → skip обрабатывается **до** intent-классификации в `process_ask`.
- `PendingRequest.skipped_slots` + хелперы `next_slot`/`slot_prompt`/`is_skip_answer` в [request_service.py](app/services/request_service.py).

### Alembic для миграций БД (`38ffe3f`)
- `alembic.ini` + `alembic/env.py` (привязан к `app.db.Base` и `DATABASE_URL`, batch для SQLite, без `fileConfig`). Baseline-миграция со всеми 13 таблицами (`alembic check` = совпадает с моделями).
- `init_db` → `_run_migrations()` с adoption: существующая БД без `alembic_version` → `stamp` baseline (не пересоздаём), иначе `upgrade head`. Удалён ручной `_migrate_user_table`. Dockerfile копирует alembic в образ.
- Будущие изменения схемы: `alembic revision --autogenerate -m "..."` → коммит миграции.

### Security-фикс eval-раннера (`032975f`)
- http-режим раннера: случайные email/пароль на прогон (нет предсказуемой учётки) + guard против не-loopback хоста (`EVAL_ALLOW_REMOTE=1` для обхода). Не влияет на прод (раннер в образ не попадает).

### Email-уведомления (`b2950d7`)
- [app/notifications.py](app/notifications.py) `EmailNotifier` — при создании заявки письмо назначенному ответственному. Env-driven (no-op без `SMTP_HOST`+`SMTP_FROM`), сбой отправки не ломает заявку, анонимность скрывает инициатора.
- Конфиг `SMTP_*` в [config.py](app/config.py), синглтон в `state.py`, вызов в `finalize_request`. Задокументировано в `.env.example` и `render.yaml`.
- **Включить на проде:** задать `SMTP_HOST` + `SMTP_FROM` (и при нужде `SMTP_USER`/`SMTP_PASSWORD`) в Render Dashboard.

## Состояние Render

Postgres-деплой подтверждён ранее (БД создана, логин admin работает, состояние переживает рестарт).

⚠️ **После пуша `38ffe3f` (Alembic) — проверить логи первого деплоя:** на существующем Postgres ожидается `stamp` baseline (создаётся таблица `alembic_version`, данные сохраняются), без ошибок миграции. Это был первый деплой с Alembic.

⚠️ Free-Postgres истекает через 30 дней с момента создания — пересоздать инстанс либо апгрейд на платный план ($7/мес Starter).

### Env vars в Render

| Ключ | Источник |
|---|---|
| `DATABASE_URL` | auto from `databases:` block |
| `LLM_API_KEY` | Groq key, вручную (sync: false) |
| `JWT_SECRET` | auto generateValue |
| `YC_API_KEY`, `YC_FOLDER_ID` | Yandex Cloud, вручную (опционально) |
| `INITIAL_ADMIN_EMAIL` / `INITIAL_ADMIN_PASSWORD` | `admin@mail.ru` / `p@ssw0rd!23`, вручную |
| `SMTP_HOST`, `SMTP_FROM`, `SMTP_USER`, `SMTP_PASSWORD` | вручную, опционально (email-уведомления) |
| `LLM_BASE_URL`, `LLM_MODEL`, `ENABLE_LLM_RERANK`, `SMTP_PORT`, `SMTP_USE_TLS` | auto из render.yaml |

## Локальное окружение

- Бэкенд: `http://127.0.0.1:8000` (SQLite `app_data.db`). Запуск: `source myenv/bin/activate && uvicorn app.main:app --host 127.0.0.1 --port 8000`
- Vite-dev: `http://127.0.0.1:5173` (proxy `/api → :8000`). Фронт: `cd frontend && npm run build` → `frontend/dist/`
- Тесты: `python -m pytest`
- Eval офлайн (без токенов): `python scripts/run_eval.py --mode local --stub-llm --cases eval/cases.json --output eval/reports/latest.json --fail-on-errors`
- Eval с реальным LLM (нужен `LLM_API_KEY`): `python scripts/run_eval.py --mode local --cases eval/intent_cases.json --output eval/reports/intent-latest.json`

## Точка возобновления

Роадмап Этапа 5 закрыт. При следующем входе:
1. Прочитать этот файл.
2. Проверить логи деплоя Render на Alembic-миграцию (см. «Состояние Render»).
3. Дождаться новой задачи от пользователя — явного следующего пункта роадмапа нет.
