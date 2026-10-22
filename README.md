# Audible AI

Turn a screenshot of your fantasy football roster into data-driven waiver-wire recommendations.
A vision model extracts the roster, a gradient-boosted model projects next-week points, and an
optimizer suggests the add/drop swaps with the highest expected gain, all validated by a
leakage-free walk-forward backtest against expert consensus.

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Repo scaffold, tooling, CI | Done |
| 2 | Data pipeline (nflverse) | In progress: weekly player stats done; schedules, rosters, injuries, snaps, Vegas lines to do |
| 3 | Features | Not started |
| 4 | Backtest harness + naive baseline | Not started |
| 5 | LightGBM model + comparison | Not started |
| 6 | Screenshot parsing | Not started |
| 7 | Optimizer | Not started |
| 8 | API and UI | Not started |
| 9 | Deployment and write-up | Not started |

No model results yet. Any results will be reported with sample sizes and bootstrapped confidence intervals.

## Layout

```
ml/          data ingestion, features, models, backtests (no web code)
  ingest/    nflverse -> Parquet loaders
  data/      gitignored Parquet output, reproducible via ml.ingest
backend/     FastAPI app
frontend/    Next.js (App Router, TypeScript, Tailwind)
tests/       pytest suite mirroring ml/ and backend/
docs/        decisions log, reports
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Node 24.

```bash
uv sync                          # Python 3.12 env from uv.lock
cp .env.example .env             # fill in keys; never commit .env
uv run pre-commit install

# Data: 2018-2025 weekly player stats -> ml/data/raw/player_stats/
uv run python -m ml.ingest.weekly_stats --start-season 2018 --end-season 2025

uv run fastapi dev backend/app/main.py     # http://localhost:8000/health
cd frontend && npm install && npm run dev  # http://localhost:3000
```

## Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy .
uv run pytest                    # offline; add `-m network` for the live nflverse test
cd frontend && npm run lint && npm run typecheck && npm test
```

CI (`.github/workflows/ci.yml`) runs all of the above and builds the backend Docker image.

## Data sources

Player stats come from [nflverse](https://github.com/nflverse) via
[nflreadpy](https://github.com/nflverse/nflreadpy). nflverse data is openly licensed; see each
dataset's license on the nflverse-data releases page.
