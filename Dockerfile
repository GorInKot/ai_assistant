# Multi-stage build: сначала собираем React-фронт (нужен Node.js), потом
# кладём готовый dist/ в Python-образ. На Render это запускается через
# runtime: docker в render.yaml — Render сам соберёт по этому Dockerfile.

# ============= Stage 1: build frontend =============
FROM node:20-alpine AS frontend-builder

WORKDIR /build

# package*.json кэшируется отдельно от исходников — пересборка npm install
# происходит только при изменении зависимостей.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ============= Stage 2: python runtime =============
FROM python:3.12-slim

# Системные пакеты для python-jose (cryptography) и passlib (bcrypt) обычно
# уже есть в slim, но на всякий случай ставим build-essential для возможных
# wheel-промахов. После установки удаляем — экономим место в image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем код приложения и базу знаний из репо.
COPY app/ ./app/
COPY knowledge_base/ ./knowledge_base/

# Готовый фронтенд из stage 1 (минимизированные JS/CSS + index.html).
COPY --from=frontend-builder /build/dist ./frontend/dist

# Render задаёт PORT через env. По умолчанию 8000 — для локальной отладки image.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
