# Audible AI

Audible AI turns a screenshot of a fantasy football roster into data-driven waiver-wire recommendations.
A vision model extracts the roster, a gradient-boosted model projects next-week points per player, and an
optimizer suggests the add/drop swaps with the highest expected gain. Results are validated with a
leakage-free walk-forward backtest against expert consensus rankings.

This is a portfolio project for ML, software engineering, and project management roles. Code quality,
reproducibility, honest evaluation, and clear documentation matter as much as model accuracy.

## Tech stack

- **ML / data:** Python 3.12, uv, Polars, scikit-learn, LightGBM, Optuna, MLflow (local), nflreadpy (nflverse)
- **Backend:** FastAPI, Pydantic v2, SQLAlchemy, Postgres, Alembic
- **Screenshot parsing:** Anthropic API (vision). Model name comes from the `ANTHROPIC_MODEL` env var, never hardcoded.
- **Optimizer:** PuLP or OR-Tools (linear programming)
- **Frontend:** Next.js (App Router), TypeScript, Tailwind, shadcn/ui, TanStack Query
- **Ops:** Docker, docker-compose, GitHub Actions (CI and weekly data refresh), Fly.io or Railway for deploy
- **Quality:** ruff, mypy, pytest, pre-commit; eslint, prettier, vitest on the frontend

## Repo layout

```
audible-ai/
  ml/              # data loading, features, training, backtesting (no web code here)
    data/          # raw and processed Parquet (gitignored; reproducible via scripts)
    features/
    models/
    backtest/
  backend/         # FastAPI app: parsing, projections, recommendations
  frontend/        # Next.js app
  docs/            # PRD, architecture diagram, backtest report, decisions
  .github/workflows/
```

## Build order (work one phase at a time)

1. Repo scaffold, tooling, CI
2. Data pipeline: nflverse weekly stats, schedules, rosters, injuries, snap counts, Vegas lines
3. Features: rolling usage, defense-vs-position (strength-of-schedule adjusted), game context
4. Backtest harness and naive baseline (last-3-weeks average)
5. LightGBM model, tuning, comparison against baseline and expert consensus
6. Screenshot parsing: image to validated roster JSON, matched to player IDs
7. Optimizer: best add/drop swaps given roster, league scoring, and free agents
8. API and UI: upload, parsed roster review, recommendations
9. Deployment, README, results write-up

Finish and commit each phase before starting the next. Do not start a later phase unless asked.

## Hard rules for ML and evaluation

- **No data leakage.** Features for week W may only use data available before kickoff of week W.
  Training for week W uses only weeks before W. Every feature function takes an explicit `as_of` week.
- **Walk-forward validation only.** Never use random train/test splits on time-series data.
- **Always compare against baselines:** last-3-weeks average and expert consensus (FantasyPros archives).
  Check the source's terms of use before scraping or storing data; prefer published or permitted exports.
- **Metrics:** RMSE/MAE on points, Spearman rank correlation by position, and points gained from
  recommended pickups versus expert-recommended pickups.
- **Report uncertainty.** Use bootstrapped confidence intervals. Never claim "#1" or "beats experts"
  without showing the sample size and interval. Weekly fantasy results are noisy; say so in docs.
- **Hold out 2026 as a live test set.** Do not tune on it.
- **Reproducibility:** set random seeds, log every run to MLflow (params, metrics, data version, git SHA).

## Coding standards

- Python: type hints everywhere, ruff and mypy clean, docstrings on public functions, small pure functions.
- Prefer Polars over pandas. No notebooks in `main` except under `docs/` for write-ups.
- Pydantic models for every API request, response, and parsed roster.
- Every pipeline step and feature gets a pytest test; leakage checks get dedicated tests.
- TypeScript strict mode. Components are small and typed, with loading and error states.
- Handle errors explicitly. The screenshot parser must return confidence flags and let the user correct players.
- Commits: small, imperative messages (e.g., "Add rolling target share feature").

## Security and secrets

- Never commit secrets. Keys live in `.env` (gitignored); keep `.env.example` up to date.
- Never print or log API keys. Validate and size-limit uploaded images; do not persist screenshots by default.
- Pin dependency versions and avoid brand-new releases when choosing packages.

## Commands

- Setup: `uv sync` and `cd frontend && npm install`
- Backend dev: `uv run fastapi dev backend/app/main.py`
- Frontend dev: `cd frontend && npm run dev`
- Tests: `uv run pytest` (add `-m network` for live nflverse tests) and `cd frontend && npm test`
- Lint and types: `uv run ruff check . && uv run ruff format --check . && uv run mypy .`
- Frontend checks: `cd frontend && npm run lint && npm run typecheck && npm test`
- Ingest all raw data: `uv run python -m ml.ingest --start-season 2018 --end-season 2025` (add `--force` to refresh)
- Ingest one dataset: `uv run python -m ml.ingest.<weekly_stats|schedules|rosters_weekly|injuries|snap_counts|players>`
- Build features: `uv run python -m ml.features.build --start-season 2018 --end-season 2025`
- Backtest: `uv run python -m ml.backtest.run --start-season 2019 --end-season 2025` (2018 is burn-in; 2026 is rejected)
- MLflow UI: `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`

(Update these as the repo takes shape.)

## How to work with me

- Use plan mode for anything touching more than two files, and show the plan before editing.
- Explain modeling and architecture choices briefly so I can defend them in interviews.
- Ask before adding a new major dependency or changing the stack.
- After each phase, update `docs/` (decisions log, README status, results) and summarize what changed.
- Flag anything that looks like leakage, overfitting, or an unsupported claim.
