# Скопируйте в env.sh и впишите секрет — тот же, что PORTAL_SSO_SECRET в .env ассистента.
# Сгенерировать: python3 -c "import secrets; print(secrets.token_hex(32))"
export ASSISTANT_SSO_SECRET=
# Адрес ассистента; пусто — тот же хост, что у портала, порт ASSISTANT_PORT.
export ASSISTANT_URL=
export ASSISTANT_PORT=8000
export PORTAL_PORT=8080
