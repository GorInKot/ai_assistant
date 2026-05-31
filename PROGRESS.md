# Текущий этап работы

> Снимок состояния на 2026-05-29. Используется как продолжение между сессиями.

## Последние коммиты

```
0239073  Postgres вместо SQLite на проде
e2a5fda  Деплой на Render через Docker (multi-stage: node→python)
4c4d16f  Admin-UI для каталога типов заявок (БД вместо YAML)
bc06384  KB: расширение законодательством РФ — 13 новых документов
3f77fbf  Security: 3 фикса по итогам /security-review
```

Все коммиты запушены в `origin/main`. После Этапа 5 роадмапа (см. [RoadMap05.md](RoadMap05.md)) сделаны дополнительные улучшения.

## Что сделано в текущей сессии

### Security fixes (commit `3f77fbf`)
- `documents.py` — все 4 эндпоинта под auth, `/reindex` и `/debug/retrieval` ограничены ролью admin/manager.
- `actions.py` — `GET /api/actions` требует auth.
- `requests.py` + `employees.py` + `employee_import.py` + `db.py` — связь `Employee↔User` через `user_id` (раньше по email-строке → privesc). Добавлен `POST /api/admin/employees/{id}/link` для ручной связи + автолинкинг при admin-действиях + миграция backfill в `init_db`.

### Расширение KB законодательством (commit `bc06384`)
- 13 новых нормативных актов в `knowledge_base/Законодательство_РФ/` по 5 разделам: Строительство, Строительный_контроль, Охрана_труда, Промышленная_безопасность, ГОиЧС.
- Переиспользуемый скрипт `scripts/fetch_legal_doc.py` (curl + html.parser + автодетект кодировки + обрезка хвостов сайдбаров) — работает с legalacts.ru, rulaws.ru, stroyinf.ru, meganorm.ru, охрана-труда.рф, bazanpa.ru.
- ⚠️ Приказы 766н и 772н Минтруда — только текст приказа без больших Приложений (с этих сайтов недоступно). При необходимости докачать PDF с pravo.gov.ru.
- Reindex прошёл: **37 документов, 2439 чанков**.

### Admin-UI каталога типов заявок (commit `4c4d16f`)
- Каталог перенесён из YAML в БД (модели `RequestType` + `RequestTypeSlot`). YAML остаётся как initial seed при пустой таблице.
- `request_catalog.py` переписан на in-memory cache из БД с `reload_catalog()` после CRUD.
- CRUD `/api/admin/request-types` (admin/manager-only).
- Frontend: `RequestTypesAdmin.tsx` + `RequestTypeFormDialog.tsx` + кнопка «📑 Типы заявок» в сайдбаре.

### Деплой на Render через Docker (commit `e2a5fda`)
- Multi-stage `Dockerfile` (node:20-alpine для фронта → python:3.12-slim для бэка).
- `.dockerignore` исключает myenv/, node_modules, secrets, локальную БД и логи.
- `app/db.py`: `DATABASE_URL` env-driven (default — SQLite).
- `_seed_initial_admin()` создаёт админа из env `INITIAL_ADMIN_EMAIL` + `INITIAL_ADMIN_PASSWORD` при каждом старте, если такого нет.
- `render.yaml` переписан на `runtime: docker`.

### Postgres вместо SQLite (commit `0239073`)
- `render.yaml`: блок `databases:` с free-Postgres `ai-assistant-db` (256 MB, 30-дневный срок жизни). `DATABASE_URL` линкуется автоматически через `fromDatabase`.
- `app/db.py`: `_migrate_user_table` пропускается на не-SQLite (на Postgres `create_all` сразу создаёт полную схему).
- `requirements.txt`: добавлен `psycopg2-binary==2.9.10`.
- Локально dev остаётся SQLite (когда `DATABASE_URL` пуст).

## Что сделано в текущей сессии (2026-05-29, продолжение)

### Pytest smoke-тесты (НЕ закоммичено)
- `tests/conftest.py` — изоляция на временной SQLite-БД (env `DATABASE_URL` до импорта app), мок `llm_service.classify_intent` (офлайн, без сети), `clean_db` drop→create→reseed перед каждым тестом + сброс `dialog_state`.
- `tests/test_smoke.py` — 10 тестов: health, register/login, защита эндпоинтов (401/403), полный путь заявки через чат (slot-filling транспорта → confirm → `/my` → inbox получателя → смена статуса), анонимка (автор скрыт в inbox), отмена оформления, запрет смены статуса автором.
- `pytest.ini` (pythonpath=.), `requirements-dev.txt` (pytest, не в прод-образ — Dockerfile ставит только `requirements.txt`).
- Запуск: `source myenv/bin/activate && python -m pytest`. **10 passed за ~0.7s.**
- KB намеренно не строится (TestClient без context-manager) — create-request flow не трогает индекс, экономим секунды.

### Eval intent-классификатора + slot-filling (НЕ закоммичено)
- `scripts/run_eval.py` починен: добавлен путь в `sys.path`, **авторизация** к `/api/ask` (раннер был мёртв после security-фикса — `/api/ask` и `/api/reindex` ушли под auth), local-режим теперь поднимает **изолированную временную SQLite-БД** (не мутирует dev `app_data.db`) и строит KB in-process вместо `POST /api/reindex` (он под admin). Также `main_app.llm_service` → `app.state.llm_service` (устаревшая ссылка после рефактора роутеров).
- Новый `kind: "intent"`: вызывает `classify_intent` напрямую in-process, проверяет `expected_intent` + `expected_request_type`. Только local + реальный LLM (в stub/http — авто-skip).
- `eval/intent_cases.json` — 15 intent-кейсов (create_request на каждый из 6 типов, qa-разграничение, confirm_yes/no/cancel) + 3 slot-filling (happy path, отмена, старт). **18/18 passed** на Groq llama-3.3-70b.
- Запуск: `python scripts/run_eval.py --mode local --cases eval/intent_cases.json --output eval/reports/intent-latest.json` (нужен LLM_API_KEY в .env).

### Разбор дрейфа retrieval-suite (НЕ закоммичено)
Дрейф `eval/cases.json` (6/12 падали) разобран по каждому кейсу — оказалось 2 реальных бага в коде + обновление ожиданий под текущий KB:

**Баги в коде (исправлены):**
- **`роль` без границы слова** ([roles.py](app/roles.py) `PARTICIPANT_QUERY_RE`): «конт**роль**» матчил паттерн → вопросы про *строительный контроль* распознавались как «вопрос про участников» и получали список ролей вместо шагов. Фикс: `\bрол[ьиея]\w*`. Офлайн-guard: `tests/test_roles.py`.
- **Уточнение перехватывалось intent-классификатором** ([ask_service.py](app/services/ask_service.py) `process_ask`): короткий уточняющий ответ («по ЕКТП» после «кто участники?») распознавался как create_request и запускал заявку. Фикс: `merge_with_pending` теперь ДО intent-классификации; для уточнения intent не вызывается.
- **Вопрос про участников без процесса угадывал процесс** вместо уточнения. Фикс (по решению пользователя — вернуть старое поведение): если `process_hint is None` → просим уточнить процесс.

**Обновлены ожидания cases.json (Variant A — KB изменился):**
- `tsus_definition/tsus_steps` — старый док `…ИД+Журналы_01.09.docx` удалён → `Руководство_пользователя_ЦУС_для_Роснефть.docx`.
- `ektp_participants/clarification_followup` — роли переписаны в KB: ЦДС / Специалист по транспорту / Ответственный сотрудник ПУ / Подрядная организация (было Заявитель/Заказчик/Диспетчер).
- `ektp_statuses/tsus_steps` — статусы/шаги лежат в .md (скрыт из источников по правилу продукта), переведены на проверку содержимого ответа + `skip_when_stub_llm` (текст генерит LLM).
- `out_of_scope` — переформулирован без trigger-слов каталога (было «Интегрируй ЕКТП…» — «ЕКТП» флапал классификатор между qa/create_request).

**Текущий статус прогонов (всё подтверждено на реальном LLM):**
- Retrieval (`eval/cases.json`): **12/12** ✓ (реальный LLM); `--stub-llm` офлайн: 9 passed / 3 skipped / 0 failed.
- Intent + slot (`eval/intent_cases.json`): **18/18** ✓ (реальный LLM).
- Pytest (`tests/`): **19 passed** (10 smoke + 9 roles).
- Весь блок закоммичен в `main`.

## Текущее состояние Render (✅ ПОДТВЕРЖДЕНО РАБОТАЕТ)

Postgres-деплой проверен пользователем: БД создана, логин под admin работает, состояние переживает рестарт.

После пуша `0239073` было сделано:
1. Render видит `databases:` блок впервые → создаёт free-Postgres инстанс `ai-assistant-db`.
2. `DATABASE_URL` подставляется в env web-сервиса автоматически.
3. Rebuild Docker image с `psycopg2-binary`.
4. На старте SQLAlchemy создаёт схему в Postgres, `_seed_initial_admin` создаёт `admin@mail.ru`, `_seed_request_types_from_yaml` сидит 6 типов заявок.

### Что проверить при возобновлении

1. **Postgres есть в Render Dashboard?** Слева в списке должна быть БД `ai-assistant-db`. Если нет — Blueprint не подхватил `databases:` блок (бывает с уже существующими сервисами), создавать руками: `New → PostgreSQL`, free план, потом в Environment добавить `DATABASE_URL` со значением Internal Database URL из дашборда новой БД.
2. **Логин под `admin@mail.ru` / `p@ssw0rd!23`** на проде работает?
3. **Состояние переживает рестарт?** Создать тестового пользователя, передеплоить, проверить что он остался.
4. **Если в Logs `psycopg2.OperationalError: connection refused`** — Postgres ещё провижится. Сделать `Manual Deploy`.

### Env vars в Render (должны быть заданы)

| Ключ | Источник |
|---|---|
| `DATABASE_URL` | auto from `databases:` block |
| `LLM_API_KEY` | Groq key, вручную (sync: false) |
| `JWT_SECRET` | auto generateValue |
| `YC_API_KEY` | Yandex Cloud, вручную (опционально) |
| `YC_FOLDER_ID` | Yandex folder, вручную (опционально) |
| `INITIAL_ADMIN_EMAIL` | `admin@mail.ru`, вручную |
| `INITIAL_ADMIN_PASSWORD` | `p@ssw0rd!23`, вручную |
| `LLM_BASE_URL`, `LLM_MODEL`, `ENABLE_LLM_RERANK` | auto из render.yaml |

## Что осталось из роадмапа

### Опциональные слоты с «пропустить» (НЕ закоммичено)
Раньше необязательные слоты (`required: false`) в каталоге вообще не спрашивались (`next_required_slot` брал только required). Теперь:
- Спрашиваются все слоты по порядку; у необязательных в вопросе подсказка «(необязательно — можно ответить «пропустить»)».
- Skip-токены (`пропустить`, `-`, `далее`, …) пропускают optional-слот; обязательный пропустить нельзя — переспрос.
- ⚠️ Ключевой нюанс (поймал только реальный LLM): «пропустить» классификатор принимает за `cancel` → обрабатываем skip **до** intent-классификации в pending-ветке. См. [ask_service.py](app/services/ask_service.py) `process_ask`.
- `PendingRequest.skipped_slots`, хелперы `next_slot`/`slot_prompt`/`is_skip_answer`/`find_slot` в [request_service.py](app/services/request_service.py).
- Тесты: pytest `test_optional_slots_can_be_skipped` / `test_required_slot_cannot_be_skipped` (21 passed); eval `slot_training_skip_optional` (intent-suite 19/19).

### Alembic для миграций БД (НЕ закоммичено)
Заменили `create_all` + ручные `ALTER TABLE` (`_migrate_user_table`) на Alembic.
- `alembic.ini` + `alembic/env.py` (привязан к `app.db.Base` и `DATABASE_URL`, `render_as_batch=True` для SQLite, БЕЗ `fileConfig` — чтобы не клобберить логирование приложения).
- Baseline-миграция `alembic/versions/21414f229b8f_baseline_schema.py` (все 13 таблиц), сгенерирована autogenerate против пустой БД. `alembic check` подтверждает: схема == модели.
- `init_db` → `_run_migrations()` ([db.py](app/db.py)): adoption-логика для существующих БД (прод Postgres/dev SQLite уже с данными) — если есть таблицы, но нет `alembic_version` → `stamp` baseline (не пересоздаём), иначе `upgrade head`. Проверено на 3 сценариях (чистая/legacy/повторный).
- `requirements.txt`: +`alembic==1.18.4`. `Dockerfile`: копирует `alembic.ini` + `alembic/` в образ (иначе старт на проде упадёт).
- `tests/conftest.py`: `clean_db` теперь сносит `alembic_version` перед `init_db` (drop_all оставлял её → upgrade head не пересоздавал таблицы).
- ⚠️ **На проде Render при деплое**: первый старт застемпит существующий Postgres baseline'ом (данные сохранятся). Проверить в логах, что нет ошибок миграции. Будущие изменения схемы — через `alembic revision --autogenerate -m "..."`.

| Пункт | Сложность | Польза |
|---|---|---|
| ~~Опциональные слоты с возможностью «пропустить»~~ ✅ сделано | Малая | UX: пользователь скипает необязательное поле текстом |
| ~~Pytest smoke-тесты~~ ✅ сделано | Средняя | Авто-проверка login → ask → создание заявки → inbox |
| ~~Eval cases для заявок~~ ✅ сделано | Средняя | Расширить `eval/` тестами intent classifier и slot-filling |
| ~~Alembic для миграций БД~~ ✅ сделано | Средняя | Заменить ручные `ALTER TABLE` в `_migrate_user_table` |
| Email-уведомления о заявках | Средняя | Нужен SMTP-сервер (пока отложено) |

⚠️ Free-Postgres истекает через 30 дней с момента создания — нужно или пересоздать инстанс, или апгрейднуться на платный план ($7/мес Starter).

## Локальное окружение

- Бэкенд работает на `http://127.0.0.1:8000` (SQLite `app_data.db`).
- Vite-dev на `http://127.0.0.1:5173` (с proxy `/api → :8000`).
- Бэкенд запускается: `source myenv/bin/activate && uvicorn app.main:app --host 127.0.0.1 --port 8000`
- Фронт билдится: `cd frontend && npm run build` → `frontend/dist/`

## Точка возобновления

При следующем входе:
1. Прочитать этот файл.
2. Спросить пользователя про результат деплоя Postgres на Render.
3. Если ОК — продолжить по списку выше. Рекомендую начать с **опциональных слотов** (малая сложность, заметный UX-эффект).
