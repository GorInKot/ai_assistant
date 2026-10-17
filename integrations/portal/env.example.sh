# Скопируйте в env.sh и впишите секрет — тот же, что PORTAL_SSO_SECRET в .env ассистента.
# Сгенерировать: python3 -c "import secrets; print(secrets.token_hex(32))"
export ASSISTANT_SSO_SECRET=
# Адрес ассистента; пусто — тот же хост, что у портала, порт ASSISTANT_PORT.
export ASSISTANT_URL=
export ASSISTANT_PORT=8000
export PORTAL_PORT=8080
# Обработка записей IVA: ffmpeg и whisper-cli должны быть доступны в PATH.
# Модель по умолчанию: tools/meeting/ggml-small.bin в проекте ai-assistant.
# export MEETING_WHISPER_MODEL=/абсолютный/путь/к/ggml-small.bin
# export MEETING_LLM_BASE_URL=http://127.0.0.1:11434/v1
# export MEETING_LLM_MODEL=ministral-3:3b
