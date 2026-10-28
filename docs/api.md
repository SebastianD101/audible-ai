# API

FastAPI app in `backend/app/`. Interactive docs are at `http://localhost:8000/docs` when it's
running. The contract is snapshotted in `tests/backend/api/openapi.json`. The frontend's types
in `frontend/lib/api.ts` mirror it.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/api/projections/meta` | Season, week, version, and row count of the projections being served |
| GET | `/api/players/search?q=&position=&limit=` | Player lookup for corrections and free agents, with this week's projection (null if none) |
| POST | `/api/parse` | Multipart `file` (PNG/JPEG/WebP/GIF, ≤5 MB) and `kind` (`roster` or `free_agents`): extracted rows, matches, review flags, and player info |
| GET | `/api/sleeper/leagues?username=` | A Sleeper user and their leagues in the projected season, with a scoring label |
| POST | `/api/sleeper/import` | Body `{league_id, user_id}`: the user's roster and the league's top free agents (both shaped like `/api/parse` responses), lineup settings, and scoring warnings |
| POST | `/api/recommendations` | Body `{roster, free_agents, locked, max_moves (1-3), settings}`: adds, drops, projected gain, lineups before and after, alternatives, protected players with reasons, caveats, backtest evidence |

## Data flow

1. **Weekly, offline:** ingest the latest nflverse data, then run `ml.serve.project`, which
   writes `ml/data/projections/season=YYYY/week=WW.parquet` and a `.json` manifest.
2. **Per request:**
   - The API loads the newest projections file, cached until the file changes.
   - It loads the season's player index from `rosters_weekly`.
   - It parses screenshots with the configured reader (Claude or local OCR, `PARSER`).
   - It imports Sleeper leagues from Sleeper's public API, with responses cached for a minute.
   - It runs `ml.optimize` for recommendations.
3. **Stored:** parse results and Sleeper imports (rows and matches only, never the image) and
   recommendation runs (request, response, and projections version). Postgres comes from
   `docker compose`, or a local SQLite file when `DATABASE_URL` is unset. The schema comes
   from Alembic migrations.

## Errors

| Status | When | What to do |
| --- | --- | --- |
| 404 | Unknown Sleeper user or league, or no team in that league | Check the username |
| 413 | Upload over 5 MB | Crop or compress the screenshot |
| 422 | Not an image; invalid settings; unknown player IDs; a Sleeper league from another season | Fix the input (the detail says which) |
| 502 | The model declined, or returned an unreadable reply | Try a clearer screenshot, or add players by search |
| 503 | No projections file; rosters not ingested; API key or model missing; rate limited or Sleeper unreachable (`Retry-After`) | Run the command in the detail, set `.env`, or retry later |

## Running it

```bash
uv run alembic -c backend/alembic.ini upgrade head   # once, and after new migrations
uv run fastapi dev backend/app/main.py               # http://localhost:8000/docs
cd frontend && npm run dev                           # http://localhost:3000
# or the API with Postgres: docker compose up --build
```
