# syntax=docker/dockerfile:1
# Backend image: FastAPI app plus the ml package it imports projections from.
# Targets:
#   runtime  code only; data is mounted (docker-compose, CI)
#   serve    the default: runtime plus this week's projections and rosters, which is what
#            Fly deploys (see docs/deploy.md). Each deploy is immutable and carries its data.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --no-install-project
# Fetch the local-OCR models (~31 MB) at build time, not on the first upload.
RUN .venv/bin/python -c "from rapidocr import RapidOCR; RapidOCR()"
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
