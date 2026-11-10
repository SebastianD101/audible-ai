# API

FastAPI app in `backend/app/`. Interactive docs are at `http://localhost:8000/docs` when it's
running. The contract is snapshotted in `tests/backend/api/openapi.json`. The frontend's types
in `frontend/lib/api.ts` mirror it.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/api/projections/meta` | Season, week, version, and row count of the projections being served |
| GET | `/api/players/search?q=&position=&limit=` | Player lookup for adding players by name and correcting rows, with this week's projection (null if none) |
| GET | `/api/sleeper/leagues?username=` | A Sleeper user and their leagues in the projected season, with a scoring label |
| POST | `/api/sleeper/import` | Body `{league_id, user_id}`: the user's roster and the league's top free agents (each a list of review rows with matches, review flags, and player info), lineup settings, and scoring warnings |
| POST | `/api/recommendations` | Body `{roster, free_agents, locked, max_moves (1-3), settings}`: adds, drops, projected gain, lineups before and after, alternatives, protected players with reasons, caveats, backtest evidence |

## Data flow

1. **Weekly, offline:** ingest the latest nflverse data, then run `ml.serve.project`, which
   writes `ml/data/projections/season=YYYY/week=WW.parquet` and a `.json` manifest.
2. **Per request:**
   - The API loads the newest projections file, cached until the file changes.
   - It loads the season's player index from `rosters_weekly`.
   - It imports Sleeper leagues from Sleeper's public API, with responses cached for a minute.
   - It runs `ml.optimize` for recommendations.
3. **Stored:** Sleeper imports (rows and matches, in the `parse_records` table, whose name predates the import) and
   recommendation runs (request, response, and projections version). Postgres comes from
   `docker compose`, or a local SQLite file when `DATABASE_URL` is unset. The schema comes
   from Alembic migrations.

## Errors

| Status | When | What to do |
| --- | --- | --- |
| 404 | Unknown Sleeper user or league, or no team in that league | Check the username |
| 422 | Invalid settings; unknown player IDs; a Sleeper league from another season | Fix the input (the detail says which) |
| 503 | No projections file; rosters not ingested; rate limited or Sleeper unreachable (`Retry-After`) | Run the command in the detail, set `.env`, or retry later |

## Running it

```bash
uv run alembic -c backend/alembic.ini upgrade head   # once, and after new migrations
uv run fastapi dev backend/app/main.py               # http://localhost:8000/docs
cd frontend && npm run dev                           # http://localhost:3000
# or the API with Postgres: docker compose up --build
```
