# Audible AI

Turn a screenshot of your fantasy football roster into data-driven waiver-wire recommendations.
A vision model extracts the roster, a gradient-boosted model projects next-week points, and an
optimizer suggests the add/drop swaps with the highest expected gain, all validated by a
leakage-free walk-forward backtest against expert consensus.

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Repo scaffold, tooling, CI | Done |
| 2 | Data pipeline (nflverse) | Done: player stats, schedules + Vegas lines, weekly rosters, injuries, snap counts, ID crosswalk (2018-2025) |
| 3 | Features | Done: 63 waiver-time features (usage, SOS-adjusted defense vs position, context, availability, vacated opportunity), leakage-tested |
| 4 | Backtest harness + naive baseline | Done: walk-forward harness, week-block bootstrap CIs, MLflow; baselines in [docs/backtest.md](docs/backtest.md) |
| 5 | LightGBM model + comparison | Done: nested-tuned LightGBM vs both baselines, kickoff-information ablation; expert consensus deferred (terms of use) |
| 6 | Screenshot parsing | Not started |
| 7 | Optimizer | Not started |
| 8 | API and UI | Not started |
| 9 | Deployment and write-up | Not started |

Results are in [docs/backtest.md](docs/backtest.md). On 2020-2025 walk-forward (101 weeks, 15,756 relevant player-weeks), LightGBM beats the stronger baseline (season average) by 1.89 RMSE [1.80, 2.01] and +0.28 weekly Spearman [0.27, 0.30]. Most of that gain comes from knowing who won't play. There's no expert-consensus comparison yet, so nothing here is a claim about beating experts. Weekly fantasy scoring is noisy, so every comparison is reported with sample sizes and paired bootstrap intervals.

## Layout

```
ml/          data ingestion, features, models, backtests (no web code)
  ingest/    nflverse -> validated Parquet loaders (one DatasetSpec per dataset)
  features/  leakage-free features as of waiver time (see docs/features.md)
  data/      gitignored Parquet output, reproducible via `python -m ml.ingest`
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

# Data: every 2018-2025 dataset -> ml/data/raw/<dataset>/
uv run python -m ml.ingest --start-season 2018 --end-season 2025
# or one dataset: python -m ml.ingest.<weekly_stats|schedules|rosters_weekly|injuries|snap_counts|players>

# Features: one row per (player, target week) -> ml/data/processed/features/
uv run python -m ml.features.build --start-season 2018 --end-season 2025

# Backtest (walk-forward, 2019-2025; 2026 is the locked holdout) -> ml/data/backtest/, MLflow
uv run python -m ml.backtest.run --start-season 2019 --end-season 2025
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

uv run fastapi dev backend/app/main.py     # http://localhost:8000/health
cd frontend && npm install && npm run dev  # http://localhost:3000
```

## Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy .
uv run pytest                    # offline; add `-m network` for the live nflverse tests
cd frontend && npm run lint && npm run typecheck && npm test
```

CI (`.github/workflows/ci.yml`) runs all of the above and builds the backend Docker image.

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
