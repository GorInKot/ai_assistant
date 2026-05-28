# Корпоративный ИИ-ассистент

Веб-приложение для сотрудников: единый чат с ассистентом, который отвечает по корпоративной базе знаний, оформляет заявки и маршрутизирует их ответственным сотрудникам.

## Возможности

### Чат и база знаний
- ChatGPT-подобный интерфейс: тёмный сайдбар с историей бесед слева, область сообщений справа.
- Беседы сохраняются в БД на пользователя; переименование/удаление через сайдбар.
- Ответы на основе RAG по локальной базе знаний (`knowledge_base/`).
- **Гибридный поиск:** лексический BM25 + векторный (Yandex Foundation Models, опционально) с фьюжном через RRF.
- В каждом ответе — блок «Источники» со ссылками **открыть / скачать** для PDF/DOCX.
- Поддержка `MD`, `PDF`, `DOCX`, `DOC` (последний — best effort).
- Honest fallback: если в KB нет точной информации — прямо об этом сообщает.

### Заявки в чате
- Пользователь пишет в чат «хочу на обучение» / «нужен транспорт на завтра» / «подать анонимное обращение».
- LLM-классификатор интента + slot-filling: ассистент пошагово опрашивает поля заявки.
- После подтверждения заявка сохраняется в БД и автоматически направляется ответственному (по области + подразделению с fallback на филиал/глобальный).
- Анонимные обращения: автор скрыт от получателя.
- Каталог типов заявок — в [app/data/request_types.yaml](app/data/request_types.yaml).

### Inbox для ответственных
- Страница «📋 Заявки» с вкладками **Входящие** / **Мои**.
- Бэйдж непрочитанных в сайдбаре (счётчик `new` в inbox, обновляется каждые 30 сек).
- Смена статуса заявки (`new` → `in_progress` → `done` / `rejected`).
- История событий по каждой заявке (создание, смена статуса, комментарии).

### Ролевая модель и админка
- Роли: `admin`, `manager`, `user`. Первый зарегистрированный пользователь получает `admin` автоматически.
- Страница «👥 Сотрудники» (для admin/manager): CRUD + импорт CSV/XLSX из корпоративной выгрузки.
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
# LLM (OpenAI-совместимый API)
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=...
LLM_MODEL=llama-3.3-70b-versatile
ENABLE_LLM_RERANK=0

# JWT (для прода — обязательно случайный секрет!)
JWT_SECRET=...

# Yandex Foundation Models для векторного поиска (опционально)
# Если оба пустые — работает только BM25.
YC_API_KEY=...
YC_FOLDER_ID=...
EMBEDDINGS_CACHE=logs/embeddings_cache.npz

# Пути
KB_ROOT=knowledge_base
LOG_FILE=logs/assistant.log
```

См. полный шаблон в [.env.example](.env.example).

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
├── llm.py                   — клиент LLM (генерация ответов + intent classifier)
├── embeddings.py            — клиент Yandex Foundation Models
├── roles.py                 — regex-ы ролей участников процесса
├── profile.py               — справочники подразделений
├── request_catalog.py       — загрузчик request_types.yaml
├── dialog_state.py          — in-memory state диалога (clarifications + pending requests)
├── actions.py               — старый MVP-журнал действий (deprecated)
├── logging_utils.py         — JSONL логгер запросов
├── data/
│   └── request_types.yaml   — каталог типов заявок (Этап 5)
├── routers/                 — HTTP-эндпоинты
│   ├── system.py            — /, /api/health, /assets mount
│   ├── auth.py              — /api/auth/{register,login}, /api/user/profile
│   ├── chat.py              — /api/ask, /api/dialog/clear
│   ├── conversations.py     — /api/conversations CRUD
│   ├── documents.py         — /api/documents, /api/files, /api/reindex
│   ├── profile.py           — /api/profile, /api/profile/password
│   ├── employees.py         — /api/admin/employees, /api/admin/areas, /api/responsibilities/lookup
│   ├── users_admin.py       — /api/admin/users
│   ├── requests.py          — /api/requests/{inbox,my,id,status}
│   └── actions.py           — /api/actions (legacy)
└── services/
    ├── ask_service.py       — главный pipeline /api/ask
    ├── request_service.py   — создание Request, slot-filling helpers
    ├── responsibilities.py  — lookup_responsible с fallback
    └── employee_import.py   — парсер CSV/XLSX

frontend/                    — Vite + React + TS + Tailwind
├── src/
│   ├── main.tsx, App.tsx
│   ├── api/                 — http-клиент, типы
│   ├── auth/                — AuthPage, store, useCurrentUser
│   ├── chat/                — ChatPage, Sidebar, MessageList, Composer, ProfileDialog, RequestsPage
│   └── admin/               — EmployeesAdmin, UsersAdmin, EmployeeFormDialog
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

## Каталог типов заявок

Файл [app/data/request_types.yaml](app/data/request_types.yaml). Каждый тип содержит:
- `type` — slug (используется в БД)
- `title` — отображаемое название
- `responsibility_area` — slug области ответственности (из таблицы `responsibility_areas`)
- `is_anonymous` — скрывать ли автора от получателя
- `trigger_keywords`, `examples` — подсказки для LLM-классификатора
- `slots` — поля для опроса (`name`, `question`, `required`)

После изменения YAML нужен рестарт бэкенда (каталог загружается на импорт модуля).

Текущие 6 типов: `training_request`, `medical_request`, `transport_request`, `tsus_access_request`, `ai_lab_request`, `anonymous_appeal`.

## Импорт сотрудников из Excel/CSV

Шаблон:
```csv
email,full_name,position,division,subdivision,phone,responsibility_areas
ivanov@company.ru,Иванов И.И.,Тренер,Филиал Уфа,ПУ Уфа,+7 999 000 00 00,training;medical
```

Поддерживаются русские заголовки: `фио`, `должность`, `филиал`, `подразделение`, `телефон`, `области`. Upsert по email — повторный импорт обновит существующих, не создаст дубликатов.

Доступные slug-и областей ответственности: `training`, `medical`, `ektp`, `tsus`, `ai_lab`, `legal` (можно расширить через `/api/admin/areas`).

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

### Сотрудники и ответственные
- `GET /api/admin/employees` — список (`q`, `division`, `subdivision`, `is_active`)
- `POST /api/admin/employees` — создать
- `PUT /api/admin/employees/{id}` — обновить
- `DELETE /api/admin/employees/{id}` — деактивировать
- `POST /api/admin/employees/import` — импорт CSV/XLSX (`multipart/form-data`)
- `GET /api/admin/areas`, `POST/DELETE` — области ответственности
- `GET /api/admin/responsibilities`, `POST/DELETE` — точечные назначения по scope
- `GET /api/responsibilities/lookup?area=...&division=...&subdivision=...` — публичный поиск ответственного

### Пользователи (admin)
- `GET /api/admin/users` — список с ролями
- `PUT /api/admin/users/{id}/roles` — изменить набор ролей
- `POST /api/admin/users/{id}/password` — сброс пароля
- `DELETE /api/admin/users/{id}` — удалить (вместе с беседами)

### Заявки
- `GET /api/requests/inbox` — заявки, назначенные на текущего юзера (через `employee.email == user.email`)
- `GET /api/requests/my` — заявки, созданные текущим юзером
- `GET /api/requests/{id}` — детали (только автор/получатель/admin)
- `PUT /api/requests/{id}/status` — сменить статус (`new`/`in_progress`/`done`/`rejected`)

### System
- `GET /api/health` — статус KB

## Логи и хранилище

- `app_data.db` — SQLite со всеми таблицами (users, conversations, messages, employees, responsibilities, requests, request_events и др.)
- `logs/assistant.log` — JSONL по каждому запросу `/api/ask`
- `logs/actions.log` — JSONL старого MVP-журнала действий (legacy, теперь заявки в БД)
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

Текущий [render.yaml](render.yaml) собирает только бэкенд (`runtime: python` без Node.js) и поэтому в проде отдаёт старый `static/index.html`.

Для деплоя нового React-UI на Render нужен либо переход на `runtime: docker` с собственным Dockerfile (`apt-get install nodejs && npm install && npm run build`), либо разнесение на два сервиса: Static Site для фронта + Web Service для бэкенда. См. комментарии в `render.yaml`.

⚠️ SQLite на Render free tier живёт на эфемерной FS — при каждом деплое БД теряется. Для прод-использования мигрировать на Postgres.

## Roadmap

См. [RoadMap05.md](RoadMap05.md) — план модернизации в 5 этапов. Все 5 этапов реализованы (коммиты от `66b154a` до `885f0e3`):

- ✅ Этап 0 — рефакторинг архитектуры + ролевое разграничение
- ✅ Этап 1 — история чатов + React UI
- ✅ Этап 2 — единый чат как production-UI
- ✅ Этап 3 — законодательство РФ + векторный поиск (Yandex)
- ✅ Этап 4 — база сотрудников + ролевая модель + Excel-импорт
- ✅ Этап 5 — заявки через чат с маршрутизацией ответственному

Открытые улучшения (вне роадмапа):
- Заполнение раздела «Законодательство РФ» полными кодексами
- Email/Telegram уведомления о новых заявках (сейчас только UI inbox)
- Admin-UI для управления каталогом типов заявок (сейчас правка YAML + рестарт)
- Alembic для миграций БД (сейчас ручной `ALTER TABLE`)
- Pytest-тесты на ключевые сценарии
- Docker-образ для деплоя на Render с Node.js
