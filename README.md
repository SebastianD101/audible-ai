# Audible AI

Turn your Sleeper fantasy football league into data-driven waiver-wire recommendations.
Sleeper's public API supplies your roster, the league's free agents, and its lineup settings (or
you add players by name), a gradient-boosted model projects next-week points, and an optimizer
suggests the add/drop swaps with the highest expected gain. Everything is evaluated with a leakage-free walk-forward
backtest against simple baselines, plus a live 2026 holdout whose projections are committed
before kickoff.

- **Live demo:** _not deployed yet; see [docs/deploy.md](docs/deploy.md)_
- **Results:** [docs/results.md](docs/results.md) (the one-page summary), with
  [backtest](docs/backtest.md), [pickups](docs/pickups.md), and [live 2026](docs/live-2026.md)
- **How it fits together:** [docs/architecture.md](docs/architecture.md) and
  [docs/decisions.md](docs/decisions.md)

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Repo scaffold, tooling, CI | Done |
| 2 | Data pipeline (nflverse) | Done: player stats, schedules + Vegas lines, weekly rosters, injuries, snap counts, ID crosswalk (2018-2025) |
| 3 | Features | Done: 63 waiver-time features (usage, SOS-adjusted defense vs position, context, availability, vacated opportunity), leakage-tested |
| 4 | Backtest harness + naive baseline | Done: walk-forward harness, week-block bootstrap CIs, MLflow; baselines in [docs/backtest.md](docs/backtest.md) |
| 5 | LightGBM model + comparison | Done: nested-tuned LightGBM vs both baselines, kickoff-information ablation; expert consensus deferred (terms of use) |
| 6 | Roster input | Done: Sleeper league import with exact nflverse IDs, the league's free agents, and lineup settings, plus add-by-name search. Screenshot parsing (Claude vision and local OCR) was built, then removed in favor of Sleeper import ([decisions](docs/decisions.md)) |
| 7 | Optimizer | Done: MILP add/drop recommender (PuLP/CBC) with a season-value drop guard, plus a historical pickup backtest ([docs/pickups.md](docs/pickups.md)); PPR only |
| 8 | API and UI | Done: weekly live projections job (2026 refits, tuning only on ≤2025), FastAPI endpoints with Postgres/Alembic persistence (never images), Next.js import → review → recommendation flow showing backtest evidence ([docs/api.md](docs/api.md)) |
| 9 | Deployment and write-up | Done: Fly.io API with the week's data baked into the image, Neon Postgres, Vercel UI; weekly GitHub Actions refresh (ingest → `--week auto` → live evaluation → commit → deploy) ([docs/deploy.md](docs/deploy.md)); results write-up in [docs/results.md](docs/results.md) |

Results are in [docs/backtest.md](docs/backtest.md). On 2020-2025 walk-forward (101 weeks, 15,756 relevant player-weeks), LightGBM beats the stronger baseline (season average) by 1.89 RMSE [1.80, 2.01] and +0.28 weekly Spearman [0.27, 0.30]. Most of that gain comes from knowing who won't play. In a simulated-league pickup backtest ([docs/pickups.md](docs/pickups.md), 5,050 roster-weeks), acting on LightGBM's one-move recommendation gains 1.98 [1.27, 2.75] real points per week, against 0.48 [−0.32, 1.37] for last-3-driven moves (paired difference +1.50 [0.36, 2.53]). There's no expert-consensus comparison yet, so nothing here is a claim about beating experts. Weekly fantasy scoring is noisy, so every comparison is reported with sample sizes and paired bootstrap intervals. Live 2026 tracking ([docs/live-2026.md](docs/live-2026.md)) started at week 8; one week is scored so far, far too few to conclude anything.

## Layout

```
ml/          data ingestion, features, models, backtests (no web code)
  ingest/    nflverse -> validated Parquet loaders (one DatasetSpec per dataset)
  features/  leakage-free features as of waiver time (see docs/features.md)
  data/      gitignored Parquet output, reproducible via `python -m ml.ingest`
  optimize/  lineup and add/drop MILP (PuLP/CBC)
  serve/     weekly live projections job and committed serving params
backend/     FastAPI app (see docs/api.md)
  app/api/      HTTP routes: projections, search, Sleeper import, recommendations
  app/db/       SQLAlchemy models; migrations in backend/alembic/
  app/roster/   player index, team codes, review-row schemas, name matching
  app/sleeper/  Sleeper API client and league import
frontend/    Next.js (App Router, TypeScript, Tailwind)
tests/       pytest suite mirroring ml/ and backend/
docs/        results, architecture, deployment, decisions log, reports
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Node 24.

```bash
uv sync                          # Python 3.12 env from uv.lock
cp .env.example .env             # optional: DATABASE_URL etc.; never commit .env
uv run pre-commit install

# Data: every 2018-2025 dataset -> ml/data/raw/<dataset>/
uv run python -m ml.ingest --start-season 2018 --end-season 2025
# or one dataset: python -m ml.ingest.<weekly_stats|schedules|rosters_weekly|injuries|snap_counts|players>

# Features: one row per (player, target week) -> ml/data/processed/features/
uv run python -m ml.features.build --start-season 2018 --end-season 2025

# Backtest (walk-forward, 2020-2025; 2026 is the locked holdout) -> ml/data/backtest/, MLflow
uv run python -m ml.backtest.run --start-season 2020 --end-season 2025
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Current-season rosters: the player index Sleeper import and search resolve IDs against
uv run python -m ml.ingest.rosters_weekly --start-season 2026 --end-season 2026

# Live projections for the upcoming week (after the previous week's games are final)
for d in weekly_stats schedules injuries snap_counts rosters_weekly; do
  uv run python -m ml.ingest.$d --start-season 2026 --end-season 2026
done
uv run python -m ml.serve.project --season 2026 --week auto   # or --week 8
uv run python -m ml.serve.evaluate --season 2026              # -> docs/live-2026.md

# App: database schema, API, UI (DATABASE_URL unset -> local SQLite file)
uv run alembic -c backend/alembic.ini upgrade head
uv run fastapi dev backend/app/main.py     # http://localhost:8000/docs
cd frontend && npm install && npm run dev  # http://localhost:3000
# or API + Postgres in Docker: docker compose up --build
```

## Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy .
uv run pytest                    # offline; add `-m network` for the live nflverse tests
cd frontend && npm run lint && npm run typecheck && npm test
```

CI (`.github/workflows/ci.yml`) runs all of the above and builds the backend Docker image.
`deploy.yml` deploys to Fly.io after CI passes on `main`, and `refresh.yml` runs the weekly
live pipeline ([docs/deploy.md](docs/deploy.md)).

## Data sources

All data comes from [nflverse](https://github.com/nflverse) via
[nflreadpy](https://github.com/nflverse/nflreadpy). nflverse data is openly licensed; see each
dataset's license on the nflverse-data releases page.

| Dataset | Grain | Used for |
| --- | --- | --- |
| `player_stats` | player-week | targets (fantasy points) and usage history |
| `schedules` | game | closing spread/total/moneyline, rest, roof; outcomes for evaluation only |
| `rosters_weekly` | player-week | team, roster status, depth chart, Sleeper/ESPN/Yahoo IDs |
| `injuries` | player-week | game designation and practice participation |
| `snap_counts` | player-game | offensive snap share (PFR IDs) |
| `players` | player (current snapshot) | ID crosswalk only (PFR to gsis, etc.) |

Each dataset directory has a `manifest.json` (rows, null IDs, kept duplicates, SHA-256 per file)
that serves as the data version. Known upstream quirks and how they're handled are in
[docs/decisions.md](docs/decisions.md).
