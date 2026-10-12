# Корпоративный ИИ-ассистент

Веб-приложение для сотрудников: единый чат с ассистентом, который отвечает по корпоративной базе знаний и работает с документами (Excel, Word, PDF).

> Оформление заявок (доступы, транспорт, обучение и др.) вынесено из ассистента: оно реализовано в корпоративном Портале, дублировать его здесь не нужно. Просьба «оформи заявку» в чате обрабатывается как обычный вопрос к базе знаний.

## Возможности

### Чат и база знаний
- ChatGPT-подобный интерфейс: тёмный сайдбар с историей бесед слева, область сообщений справа.
- Беседы сохраняются в БД на пользователя; переименование/удаление через сайдбар.
- Ответы на основе RAG по локальной базе знаний (`knowledge_base/`).
- **Гибридный поиск:** лексический BM25 + векторный (Yandex Foundation Models, опционально) с фьюжном через RRF.
- В каждом ответе — блок «Источники» со ссылками **открыть / скачать** для PDF/DOCX.
- Поддержка `MD`, `PDF`, `DOCX`, `DOC` (последний — best effort).
- Honest fallback: если в KB нет точной информации — прямо об этом сообщает.

### Файлы в чате
Файлы прикладываются к сообщению (скрепка у поля ввода), обработка идёт в памяти, на сервере файлы не сохраняются.
- **Excel (.xlsx):** показать содержимое, сравнить два файла (в том числе бланки), объединить несколько с одинаковой структурой.
- **Word (.docx, .doc):** выжимка, сравнение двух версий.
- **PDF → Word/Excel:** текст и таблицы в порядке страницы; сканы распознаются через OCR (Tesseract) с предупреждением.
- **PDF → реестр (Excel):** несколько документов (акты, счета, письма; до 10 за раз) собираются в одну таблицу по заданным полям: «составь реестр: номер, дата, подрядчик, сумма». Без перечня полей берётся стандартный набор. Каждое значение проверяется по тексту документа: чего в нём нет, подсвечивается жёлтым и попадает в колонку «Проверить»; если поле не найдено, ячейка остаётся пустой, а не заполняется догадкой.

### Ролевая модель и админка
- Роли: `admin`, `manager`, `user`. Первый зарегистрированный пользователь получает `admin` автоматически.
- Страница «🔑 Пользователи» (для admin): таблица всех юзеров, переключаемые чипы ролей, сброс пароля, удаление.
- Самозащита: нельзя снять admin/удалить самого себя.
- Смена своего пароля через окно профиля.

### Аутентификация
- JWT-токены (60 минут).
- Все API-эндпоинты (кроме `/api/auth/*` и `/api/health`) требуют токен.
- `session_id` диалога префиксуется `u{user_id}:` — нельзя перехватить чужой контекст уточнения.

## Стек

| Слой | Технологии |
|---|---|
| Backend | Python 3.10+, FastAPI, SQLAlchemy, SQLite |
| Frontend | Vite + React + TypeScript + Tailwind CSS |
| LLM | OpenAI-совместимый API (по умолчанию Groq, можно Ollama) |
| Embeddings | Yandex Foundation Models (опционально) |
| RAG | BM25 + cosine, RRF fusion, кэш эмбеддингов на диск |
| Auth | passlib (pbkdf2), python-jose (JWT) |
| Import | openpyxl (Excel), стандартный csv |

## Быстрый старт

### Требования
- Python 3.10+
- Node.js 20+ и npm
- (Опционально) API-ключ Groq для генерации ответов
- (Опционально) API-ключ Yandex Cloud для векторного поиска

### Установка
```bash
# Бэкенд
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Заполните .env (см. ниже)

# Фронтенд (один раз)
cd frontend && npm install && npm run build && cd ..
```

### Запуск (production-режим)
```bash
uvicorn app.main:app
```
Открыть `http://127.0.0.1:8000` — отдаётся собранный React-UI из `frontend/dist`.

### Запуск (dev-режим с HMR)
В двух терминалах:
```bash
# Терминал 1: backend
uvicorn app.main:app --reload

# Терминал 2: frontend (HMR на :5173, /api/* проксируется на :8000)
cd frontend && npm run dev
```
Открыть `http://127.0.0.1:5173`.

### Старый UI (deprecated)
Если `frontend/dist` отсутствует, бэкенд автоматически отдаст старый `static/index.html` со вкладками — режим только для сравнения, новые фичи туда не добавляются.

## Переменные окружения (.env)

```bash
# LLM (OpenAI-совместимый API). Локальный вариант (Ollama) — основной:
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=
LLM_MODEL=ministral-3:3b
LLM_REASONING_EFFORT=none   # отключает «размышления» у моделей, которые их умеют (Qwen3.5 и т.п.)
ENABLE_LLM_RERANK=0         # LLM-rerank на небольших локальных моделях слишком медленный
# Облачный вариант (например, Groq): LLM_BASE_URL=https://api.groq.com/openai/v1,
# LLM_API_KEY=..., LLM_MODEL=llama-3.3-70b-versatile, LLM_REASONING_EFFORT пустой.

# JWT (для прода — обязательно случайный секрет!)
JWT_SECRET=...

# Векторный поиск (опционально). Если ничего не задано — работает только BM25.
# Вариант 1 (приоритетный): локальные эмбеддинги через Ollama / LM Studio / vLLM.
EMBEDDINGS_BASE_URL=http://127.0.0.1:11434/v1
EMBEDDINGS_MODEL=qwen3-embedding:0.6b
EMBEDDINGS_QUERY_PREFIX=Instruct: Given a user question about corporate processes, retrieve passages that answer it\nQuery: 
# Вариант 2: Yandex Foundation Models (облако)
YC_API_KEY=...
YC_FOLDER_ID=...
EMBEDDINGS_CACHE=logs/embeddings_cache.npz

# Пути
KB_ROOT=knowledge_base
LOG_FILE=logs/assistant.log
```

См. полный шаблон в [.env.example](.env.example).

### Локальные модели (без внешних API)
Установить [Ollama](https://ollama.com) (`brew install ollama`), запустить `ollama serve` и скачать модель эмбеддингов:
```bash
ollama pull qwen3-embedding:0.6b   # 639 МБ, размерность 1024
```
В `.env` задать `EMBEDDINGS_BASE_URL` и `EMBEDDINGS_MODEL` (см. выше) и перезапустить бэкенд. Первый запуск считает векторы всей KB (на M1 ≈ 12 минут для ~2400 фрагментов) и кэширует их на диск; кэш привязан к модели, при смене модели пересчитывается. Данные не покидают машину.

Для генерации ответов тот же Ollama подходит как LLM (OpenAI-совместимый API): `ollama pull ministral-3:3b`, затем `LLM_BASE_URL=http://127.0.0.1:11434/v1`, `LLM_MODEL=ministral-3:3b`, `LLM_REASONING_EFFORT=none`, `ENABLE_LLM_RERANK=0`.

Проверено на MacBook M1 / 8 ГБ ОЗУ, `eval/cases.json` (12 кейсов, реальная генерация):

| Модель | В памяти | Eval | Время на 12 кейсов | Замечания |
|---|---|---|---|---|
| `ministral-3:3b` | 2,5 ГБ | **12/12** | 11 мин (8–52 с на ответ) | рекомендуется |
| `qwen3.5:4b` | 4,3 ГБ | 10/12 | 25 мин | 3 ошибки нехватки памяти (OOM) |
| `qwen3.5:2b` | ~2,7 ГБ | 6/12 | 4 мин | часто отвечает «нет информации» |

Две модели в одном Ollama (эмбеддинги + LLM) на 8 ГБ работают, но запас по памяти небольшой: на время работы стоит закрывать тяжёлые приложения. Эмбеддинги: `qwen3-embedding:0.6b` (на этом eval — 9/0 против 8/1 у BM25).

### Получение Yandex-ключей (для векторного поиска)
1. Зарегистрироваться на https://aistudio.yandex.cloud/platform/
2. Кнопка «Создать API-ключ» в правом верхнем углу
3. `folder_id` — в URL вашего каталога: `console.yandex.cloud/folders/b1g...`
4. Положить оба значения в `.env` и перезапустить бэкенд

При старте в логе появится `Yandex embedder enabled (folder_id=...)`. При первом запуске считаются эмбеддинги всех чанков KB и кэшируются на диск.

## Структура проекта

```
app/
├── main.py                  — точка входа FastAPI, подключение роутеров
├── state.py                 — синглтоны (kb_index, llm_service, embedder, dialog_state)
├── schemas.py               — pydantic-модели запросов/ответов
├── db.py                    — модели SQLAlchemy + init/seed
├── auth.py                  — JWT, require_role, assign_initial_role
├── config.py                — загрузка .env
├── kb.py                    — индекс KB (BM25 + cosine + RRF)
├── llm.py                   — клиент LLM (генерация ответов, rerank, выжимка)
├── embeddings.py            — клиент Yandex Foundation Models
├── roles.py                 — regex-ы ролей участников процесса
├── profile.py               — справочники подразделений
├── dialog_state.py          — in-memory state диалога (уточнения к вопросам)
├── actions.py               — старый MVP-журнал действий (deprecated)
├── logging_utils.py         — JSONL логгер запросов
├── routers/                 — HTTP-эндпоинты
│   ├── system.py            — /, /api/health, /assets mount
│   ├── auth.py              — /api/auth/{register,login}, /api/user/profile
│   ├── chat.py              — /api/ask, /api/dialog/clear
│   ├── conversations.py     — /api/conversations CRUD
│   ├── documents.py         — /api/documents, /api/files, /api/reindex
│   ├── profile.py           — /api/profile, /api/profile/password
│   ├── users_admin.py       — /api/admin/users
│   └── actions.py           — /api/actions (legacy)
└── services/
    ├── ask_service.py       — главный pipeline /api/ask
    ├── chat_documents.py    — файлы в чате: Excel/Word/PDF (сравнение, выжимка, конвертация)
    └── document_processing.py — ядро обработки документов

frontend/                    — Vite + React + TS + Tailwind
├── src/
│   ├── main.tsx, App.tsx
│   ├── api/                 — http-клиент, типы
│   ├── auth/                — AuthPage, store, useCurrentUser
│   ├── chat/                — ChatPage, Sidebar, MessageList, Composer, ProfileDialog
│   └── admin/               — UsersAdmin
└── dist/                    — собранный prod-bundle (после npm run build)

knowledge_base/              — база знаний
├── ЦУС_Строительный_контроль/
├── ЕКТП_Транспорт/
├── Обучение_и_медосмотр/
├── Лаборатория_ИИ/
├── Законодательство_РФ/     — добавлен на Этапе 3
│   └── Трудовое_право/      — TODO: Охрана_труда, Промбезопасность, ГОиЧС
└── glossary.md

static/index.html            — старый UI (fallback)
RoadMap05.md                 — план модернизации в 5 этапов
eval/                        — фреймворк качества RAG
scripts/run_eval.py
```

## База знаний

Поддерживаемые форматы: `PDF`, `DOC`, `DOCX`, `MD`.

Структура: каждая папка верхнего уровня в `knowledge_base/` — отдельный процесс. Внутри рекомендуется `инструкции/`, `чеклисты/`, `faq.md`.

После добавления файлов:
```bash
curl -X POST http://127.0.0.1:8000/api/reindex \
  -H "Authorization: Bearer $TOKEN"
```

### Рекомендации
- Лучший retrieval — для `PDF/DOCX` с корректно извлекаемым текстом.
- Для `DOC` используется best-effort (если извлечение ненадёжно, файл не участвует в retrieval, но остаётся доступным как источник).
- `md` — для FAQ/глоссариев/правил. В блоке «Источники» в UI они скрыты (показываются только прикладные документы).
- Имена файлов: формы — `Форма_...docx`, инструкции — `Инструкция_...pdf`.

### Векторный поиск
Если в `.env` заданы `YC_API_KEY` и `YC_FOLDER_ID`, при `build()` индекса считаются эмбеддинги всех чанков (модель `text-search-doc`) и кэшируются в `logs/embeddings_cache.npz`. При reindex пересчитываются только новые/изменённые чанки (по `hash(chunk.text)`).

Запрос пользователя эмбеддится через `text-search-query`, считается cosine top-k, результаты сливаются с BM25 через RRF (`k=60`). Score в `RetrievalResult` остаётся в шкале BM25 (для корректной работы порогов `is_context_strong`), порядок — по RRF.

## API (основное)

### Auth
- `POST /api/auth/register` — регистрация (первый юзер → admin)
- `POST /api/auth/login` — получить JWT
- `GET /api/user/profile` — данные текущего пользователя + список ролей

### Чат и история
- `POST /api/ask` — задать вопрос (с опциональным `conversation_id` для сохранения в историю)
- `POST /api/dialog/clear` — сбросить состояние диалога
- `GET /api/conversations` — список бесед
- `POST /api/conversations` — создать беседу
- `GET /api/conversations/{id}` — детали + сообщения
- `PATCH /api/conversations/{id}` — переименовать
- `DELETE /api/conversations/{id}` — удалить (каскадно с сообщениями)

### KB
- `GET /api/documents?q=...&process=...` — список документов
- `POST /api/reindex` — переиндексация
- `GET /api/debug/retrieval?q=...` — отладка ranking
- `GET /api/files/{path}` — открыть/скачать (`?download=1`)

### Профиль
- `GET /api/profile` — текущий профиль + опции подразделений
- `POST /api/profile` — обновить
- `POST /api/profile/password` — сменить свой пароль

### Пользователи (admin)
- `GET /api/admin/users` — список с ролями
- `PUT /api/admin/users/{id}/roles` — изменить набор ролей
- `POST /api/admin/users/{id}/password` — сброс пароля
- `DELETE /api/admin/users/{id}` — удалить (вместе с беседами)

### System
- `GET /api/health` — статус KB

## Логи и хранилище

- `app_data.db` — SQLite со всеми таблицами (users, roles, conversations, messages и др.)
- `logs/assistant.log` — JSONL по каждому запросу `/api/ask`
- `logs/actions.log` — JSONL старого MVP-журнала действий (legacy)
- `logs/embeddings_cache.npz` — кэш эмбеддингов для KB (не коммитится)

## Локальный eval (качество RAG)

Быстрый прогон без сети:
```bash
PYTHONPATH=. myenv/bin/python scripts/run_eval.py \
  --mode local --stub-llm --output eval/reports/latest.json --fail-on-errors
```

Baseline:
```bash
PYTHONPATH=. myenv/bin/python scripts/run_eval.py \
  --mode local --stub-llm --output eval/reports/baseline-local.json
```

Подробности — в [eval/README.md](eval/README.md).

## Деплой (Render)

Проект собирается на Render как Docker-сервис: multi-stage [Dockerfile](Dockerfile) сначала строит React-фронт в Node-образе, потом кладёт `frontend/dist/` в Python-образ с FastAPI. Конфигурация в [render.yaml](render.yaml).

### Первый деплой

1. Render Dashboard → **New → Blueprint** → выбрать этот репозиторий.
2. В разделе **Environment** задать секреты (помечены `sync: false`):
   - `LLM_API_KEY` — Groq API key.
   - `YC_API_KEY`, `YC_FOLDER_ID` — Yandex Cloud Foundation Models (для hybrid retrieval; без них работает только BM25, но быстрее cold start).
   - `INITIAL_ADMIN_EMAIL`, `INITIAL_ADMIN_PASSWORD` — pre-seeded админ. Создаётся при каждом старте, если не существует, и закрывает дыру «первый зарегистрировавшийся = admin».
   - `JWT_SECRET` Render сгенерирует автоматически (`generateValue: true`).
3. После первого билда открыть `https://<your-service>.onrender.com` и залогиниться под `INITIAL_ADMIN_*`.

### База данных

`render.yaml` определяет блок `databases:` с free-Postgres `ai-assistant-db` (256 MB, 30-дневный срок жизни инстанса). Render автоматически подставляет connection string в env-переменную `DATABASE_URL`, и при первом деплое схема создаётся через SQLAlchemy `create_all`.

Это решает главную проблему free-плана: БД (пользователи, история чатов) переживает деплои и cold start. Локально остаётся SQLite (когда `DATABASE_URL` пуст, `app/db.py` использует `sqlite:///./app_data.db`).

Через 30 дней free-Postgres истекает — нужно пересоздать инстанс или апгрейднуться на платный план ($7/мес для Starter с persistent storage).

### Ограничения free-плана

- ⚠️ Файловая система эфемерная — теряются между деплоями/cold-стартами:
  - Embeddings cache (`logs/embeddings_cache.npz`) — пересчёт ~3 мин через Yandex API.
  - Логи запросов и действий (`logs/*.log`).
- БД (Postgres) — НЕ эфемерная, переживает деплои.
- Pre-seeded admin (`INITIAL_ADMIN_*`) создаётся только если в БД нет пользователя с таким email — после первого старта это no-op.
- Чтобы ускорить cold start: закоммитить `logs/embeddings_cache.npz` в репо (попадёт в Docker image), убрав его из `.gitignore` и `.dockerignore`.

### Прод-апгрейд (опционально)

Платный план Starter ($7/мес) + persistent disk 1 GB ($1/мес) добавляет персистентность логам и embeddings-кэшу:

1. В [render.yaml](render.yaml) раскомментировать блок `disk:`.
2. В Environment добавить:
   - `LOG_FILE=/data/logs/assistant.log`
   - `EMBEDDINGS_CACHE=/data/logs/embeddings_cache.npz`
3. `DATABASE_URL` менять не нужно — Postgres уже отделён.

## Roadmap

См. [RoadMap05.md](RoadMap05.md) — план модернизации в 5 этапов. Все 5 этапов реализованы (коммиты от `66b154a` до `885f0e3`):

- ✅ Этап 0 — рефакторинг архитектуры + ролевое разграничение
- ✅ Этап 1 — история чатов + React UI
- ✅ Этап 2 — единый чат как production-UI
- ✅ Этап 3 — законодательство РФ + векторный поиск (Yandex)
- ✅ Этап 4 — база сотрудников + ролевая модель + Excel-импорт _(сотрудники и области ответственности удалены: перенесены в Портал)_
- ✅ Этап 5 — заявки через чат с маршрутизацией ответственному _(удалено: заявки оформляются в Портале)_

Открытые улучшения (вне роадмапа):
- Alembic для миграций БД (сейчас ручной `ALTER TABLE`)
- Pytest-тесты на ключевые сценарии
