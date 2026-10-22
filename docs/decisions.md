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
