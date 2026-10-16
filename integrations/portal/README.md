# Ассистент как раздел портала

Ассистент открывается плиткой «ИИ-ассистент» на главной странице корпоративного портала
(`portalrnsk`, Django) — внутри портала, без второго входа. Файлы портала не меняются:
надстройка `portal_bridge` подключается при запуске через `DJANGO_SETTINGS_MODULE`.

## Как это работает

1. Пользователь входит в портал и нажимает плитку «ИИ-ассистент» (`/assistant/`).
2. Портал выдаёт одноразовый пропуск на 60 секунд: логин, ФИО, роль; подпись — HMAC-SHA256
   общим секретом. Пропуск уходит POST-запросом прямо в окно ассистента (iframe) — в адресную
   строку и журналы не попадает.
3. Ассистент (`POST /sso/portal`) проверяет подпись, срок и одноразовость, находит или заводит
   пользователя `<логин>@portal.local` и открывает чат.

Роль берётся из портала при каждом входе: «Админ» → admin, «Руководитель» → manager, остальные —
user. Пропуском нельзя войти в учётную запись, заведённую в ассистенте напрямую, а адреса
`@portal.local` нельзя зарегистрировать. Встраивать ассистент в iframe разрешено только порталу
(`frame-ancestors` из `PORTAL_ORIGINS`).

## Что входит

| Файл | Назначение |
|---|---|
| `portal_bridge/settings.py` | настройки портала как есть + раздел ассистента; данные демо-портала — в `.data/` |
| `portal_bridge/urls.py` | адреса портала как есть + `/assistant/` |
| `portal_bridge/views.py` | плитка на главной и страница с окном ассистента |
| `portal_bridge/sso_pass.py` | пропуск (без Django; формат сверяется тестом `tests/test_portal_sso.py`) |
| `run_portal.sh` | миграции демо-базы и запуск портала на `127.0.0.1:8080` |

## Запуск на Mac

```sh
# портал: окружение Python 3.12 в его папке (один раз)
python3.12 -m venv ../PROJFORTERMINAL/.venv
../PROJFORTERMINAL/.venv/bin/pip install "Django>=4.2,<5.0" Pillow fastapi "uvicorn[standard]" \
    openpyxl python-multipart xlrd lxml python-docx

# общий секрет: integrations/portal/env.sh (ASSISTANT_SSO_SECRET) и .env ассистента:
#   PORTAL_SSO_SECRET=<тот же секрет>
#   PORTAL_ORIGINS=http://127.0.0.1:8080 http://localhost:8080
cp integrations/portal/env.example.sh integrations/portal/env.sh

integrations/portal/run_portal.sh          # портал: http://127.0.0.1:8080
myenv/bin/python -m uvicorn app.main:app   # ассистент: http://127.0.0.1:8000
```

Администратор демо-портала: `PYTHONPATH=integrations/portal:../PROJFORTERMINAL
DJANGO_SETTINGS_MODULE=portal_bridge.settings PORTAL_DATA_DIR=integrations/portal/.data
../PROJFORTERMINAL/.venv/bin/python ../PROJFORTERMINAL/manage.py createsuperuser`.

## Перенос в сам портал

Когда раздел войдёт в портал штатно: `views.assistant` и шаблон `assistant/index.html` — в
портал, плитку — в список `panels` в `portalrnsk/views.py`, модуль `assistant` — в
`ModuleAccess.MODULE_CHOICES` (тогда доступ к разделу настраивается в админке по ролям).
