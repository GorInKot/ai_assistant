#!/bin/sh
# Запуск портала portalrnsk с разделом «ИИ-ассистент» и кнопкой ассистента на всех страницах.
# Файлы портала не меняются: надстройка подключается через DJANGO_SETTINGS_MODULE и
# portal_bridge.asgi (кнопка), данные демо-портала — в .data/.
#
#   PORTAL_DIR=путь/к/порталу ./run_portal.sh     (по умолчанию — PROJFORTERMINAL рядом с проектом)
#
# Секрет и адреса — в env.sh рядом со скриптом (не в git), пример — env.example.sh.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
PORTAL_DIR=${PORTAL_DIR:-"$HERE/../../../PROJFORTERMINAL"}
PORTAL_DIR=$(cd "$PORTAL_DIR" && pwd)
[ -f "$HERE/env.sh" ] && . "$HERE/env.sh"

export DJANGO_SETTINGS_MODULE=portal_bridge.settings
export PYTHONPATH="$HERE:$PORTAL_DIR"
export PORTAL_DATA_DIR=${PORTAL_DATA_DIR:-"$HERE/.data"}
PYTHON="$PORTAL_DIR/.venv/bin/python"

for db in default metrology protocol tkzaii; do
  "$PYTHON" "$PORTAL_DIR/manage.py" migrate --noinput --database "$db" >/dev/null
done

cd "$PORTAL_DIR"
exec "$PYTHON" -m uvicorn portal_bridge.asgi:application --host 127.0.0.1 --port "${PORTAL_PORT:-8080}"
