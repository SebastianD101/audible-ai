# syntax=docker/dockerfile:1
# Backend image: FastAPI app plus the ml package it imports projections from.
# Targets:
#   runtime  code only; data is mounted (docker-compose, CI)
#   serve    the default: runtime plus this week's projections and rosters and the static
#            frontend, which is what Render deploys (see docs/deploy.md). Each deploy is
#            immutable and carries its data.

# Static export of the Next.js UI. An empty API URL makes its requests same-origin.
FROM node:24-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
ENV NEXT_PUBLIC_API_URL="" NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --no-install-project
COPY backend ./backend
COPY ml ./ml

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
CMD ["fastapi", "run", "backend/app/main.py", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS serve
COPY --chown=app:app ml/data/projections ./ml/data/projections
COPY --chown=app:app ml/data/raw/rosters_weekly ./ml/data/raw/rosters_weekly
COPY --from=frontend --chown=app:app /frontend/out ./static
ENV STATIC_DIR=/app/static
# Render's free tier has no pre-deploy step, so migrations run at startup. With DATABASE_URL
# unset that's a SQLite file inside the container: the audit log resets on every restart.
# Render sets PORT; 8000 elsewhere.
CMD ["sh", "-c", "alembic -c backend/alembic.ini upgrade head && exec fastapi run backend/app/main.py --host 0.0.0.0 --port \"${PORT:-8000}\""]
