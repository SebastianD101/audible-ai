# Decisions log

Short, dated records of choices worth explaining. Newest at the bottom.

## 2026-10-22: Single uv project for `ml` and `backend`

One `pyproject.toml` / `uv.lock` at the root (non-packaged, `package = false`). The backend
imports projection code from `ml`, so they share one pinned dependency set; splitting into a
uv workspace can wait until the dependency sets diverge (e.g. if the API image needs to drop
training-only deps like Optuna/MLflow).

## 2026-10-22: Dependency pinning policy

All Python and npm dependencies are pinned exactly, at the newest release that was at least
~2 weeks old at the time of pinning. Brand-new major versions are skipped: Polars 2.0.0
(released 2026-10-06) was passed over for 1.44.2, and Next 16.4 (2026-10-06) for 16.3.8.
`@types/node` was moved to 24.x to match the Node 24 runtime (the scaffold's 20.x conflicted
with Vitest 5's peer range).

## 2026-10-22: Raw data layer: one Parquet file per season

`ml/data/raw/player_stats/season=YYYY.parquet` plus a `manifest.json` (row counts, SHA-256 per
file, nflreadpy version, fetch time).

- **Why per season:** the weekly refresh only rewrites the current season; reads use
  `pl.scan_parquet("…/*.parquet")` with predicate pushdown.
- **Manifest as data version:** MLflow runs will log the manifest hash so every result is
  traceable to exact input files.
- **Raw stays raw:** regular season and postseason rows are both stored (`season_type`).
  About 21–22 rows per season have a null `player_id`. These are unattributed stat lines
  with 0 fantasy points; they are kept, logged and counted in the manifest, and filtered in
  the feature layer.
- **Validation at ingest:** schema and dtypes, season match, week range (17 regular-season
  weeks through 2020, 18 from 2021, plus 4 postseason weeks), and no duplicate
  `(player_id, season, week, season_type)`. Invalid data raises and is never written.
- **2026 holdout:** the loader's default range ends at 2025.

## 2026-10-22: Known approximation: stat corrections

nflverse serves *final* stats, including NFL stat corrections that usually post the Tuesday or
Wednesday after a game. A feature for week W built from week W-1 stats may therefore use
slightly corrected numbers that would have been available a few days before kickoff of W,
rather than the raw Sunday-night box score. This is within the "available before kickoff"
rule, but it is not a perfect as-of snapshot. Logged here so the backtest write-up can state it.

## 2026-10-22: One declarative spec per dataset (`ml.ingest.common`)

Every per-season loader is a `DatasetSpec` (fetch function, required columns and dtypes, key and
ID columns, how postseason is tagged, extra checks). Validation, atomic writes, skip-if-exists,
and the manifest live in one place, so each new dataset is ~60 lines and gets the same
guarantees. `uv run python -m ml.ingest` runs all of them. A failing dataset is logged and the
rest still run, and the exit code is non-zero.

Manifests now record `null_id_rows` and `duplicate_key_rows` per season. Duplicate keys are a
hard error by default. Two datasets opt into "keep and count" for documented upstream reasons
(below).

## 2026-10-22: Vegas lines come from nflverse schedules (closing lines)

`load_schedules` already carries `spread_line`, `total_line`, moneylines, and odds, with no
nulls for played games 2018-2025, so no separate odds source or scraping is needed.

- **Sign convention (verified):** positive `spread_line` = home team favored by that many
  points, the opposite of sportsbook notation. Correlation with the actual home margin is
  0.45 over 2018-2025, and a network test pins the sign.
- **Closing, not opening:** these are lines at kickoff. See the timing caveat below.
- **Outcome columns:** schedules also contain `home_score`, `away_score`, `result`, `total`,
  `overtime`. They are for targets and evaluation only. Phase 3 will add a leakage test asserting
  no feature for week W reads them for week W or later.

## 2026-10-22: Upstream data quirks found during ingest

Found by running the loaders over every season, not just spot checks.

| Dataset | Quirk | Handling |
| --- | --- | --- |
| injuries | `season`/`week` are Float64 in 2018-2020 | Cast to Int32 at ingest (verified lossless); the only dtype change made to raw data |
| injuries | 2025 drops `date_modified`, adds `season_type` | Kept as delivered; read across seasons with `pl.scan_parquet(..., missing_columns="insert")` |
| injuries | 2 rows in 2024 wk 15 repeat a player-week with a changed designation (Questionable then Out) | Kept and counted; feature layer resolves (likely most severe) |
| injuries | `practice_status` sometimes whitespace instead of null | Left as delivered; cleaned in features |
| rosters_weekly | 2019 wk 2-17: Isaiah Searight listed under Quinnen Williams's gsis ID `00-0035718` (his own is `00-0035478`) | Kept and counted; joins must dedupe |
| rosters_weekly | 2-38 null `gsis_id` rows per season | Kept and counted |
| snap_counts | `st_pct` reaches 1.01 (2021, rounding) | Not checked; only `offense_pct` is range-checked |

## 2026-10-22: Player ID crosswalk stores IDs only

Snap counts are keyed by Pro Football Reference ID. Weekly rosters map only 56-66% of
offensive snap rows to gsis IDs, while `load_players()` maps 99.65-100% per season
(2018-2025). `load_players()` is a *current* snapshot, though, and its team, status, and
position describe today. Using them for a past week would leak the future. The loader therefore stores only ID
columns and `display_name`, which makes that leak structurally impossible, and enforces unique
`gsis_id` and `pfr_id` so joins cannot fan out.

## 2026-10-22: Open question for Phase 3: what was known at waiver time?

Closing lines and Friday injury designations are known before kickoff, so they satisfy the
no-leakage rule for a *projection*. Waiver claims, however, typically process Tuesday or
Wednesday, before either is known. A backtest of *pickup decisions* that uses them would be
optimistic. Phase 3 has to choose between (a) Wednesday-available proxies (prior-week
designations and practice status, look-ahead lines where available) and (b) closing values with
the backtest write-up stating the bias. Roster `status` has no published snapshot time and
carries the same question.

## 2026-10-22: Weekly data refresh workflow deferred

Raw data is gitignored, so a scheduled GitHub Action needs somewhere to put it (artifact,
release asset, or bucket). That choice depends on where the API runs, so it moves to Phase 9.
