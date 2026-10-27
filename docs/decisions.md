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

## 2026-10-22: Information cutoff is waiver time, not kickoff

Features for target week W use only data from weeks before W, as known on the Tuesday night
after week W-1 when waiver claims process. Week-W closing lines, injury designations, and
practice reports are excluded; the published schedule (opponent, venue, rest) is allowed
because it's released in May.

- **Why:** the product recommends waiver pickups, so the backtest should only use what a
  manager knows when making that decision. A model that quietly used Friday injury news or
  closing lines would look better than it could ever be in use.
- **What it buys:** one leakage rule (`(season, week) < W`) for every dataset, enforced by
  `ml.features.asof.available_before` and tested by corrupting every future row.
- **Cost:** the strongest pre-game signal, the Vegas total, isn't available. The stand-in
  (points for/against before W) shows almost no signal. A Phase 5 "kickoff" ablation will add
  week-W lines and designations to measure how much late-week information is worth.

## 2026-10-22: Spine and labels

- **Spine for week W (W >= 2):** fantasy-position players on a 53-man or reserve list in week
  W-1 (ACT/INA/RES) or with a stat line earlier this season. Bye-week rows are dropped, and
  week 1 is excluded (no in-season data and no prior-week roster yet).
- **Coverage (2018-2025):** 99.48% of week-2+ PPR points. Most of the misses are debut starts
  (e.g. Will Levis, 2023 week 8) that nobody could have rostered at waiver time.
- **Practice squad and cut players** enter only through a stat line. Including them would make 58%
  of rows trivial zeros instead of 44%, for +0.4 points of coverage. Trivial zeros flatter RMSE,
  so Phase 4 must also report metrics on a fantasy-relevant subset.
- **Label = 0 when the team played and the player didn't.** A pickup who sits is worth nothing, so
  the target includes availability risk.
- **Labels are named `label_*`.** The first draft used `target_*`, which collided with the
  `target_share` feature. A guard that filtered "target*" columns would have dropped real
  features, and one that only filtered exact names would miss future collisions.

## 2026-10-22: SOS-adjusted defense vs position

For each game a defense played before W, the residual is the opposing offense's points at a
position minus that offense's average at the position in its other games before W
(leave-one-out). The rating is the mean residual, shrunk toward 0.5 times the prior-season
rating with K = 4 games. K and the 0.5 carry-over are fixed choices, not tuned. Controlling for
player level, it correlates more strongly with the outcome than raw points allowed at all
four positions (see `docs/features.md`).

## 2026-10-22: Vacated opportunity counts only new absences

The first version counted any teammate who missed the last game or was on reserve. That
flagged 37% of WR/TE rows (season-long IR roles were redistributed long ago) and carried
almost no signal: flagged WR/TE beat their season average by +0.14 PPG, vs +0.01 for the rest.
Counting only teammates who played the game before last, or left the last game early, flags
14% of WR/TE rows; those beat their season average by +0.21 (vs +0.03). For RBs with newly
vacated rush share above 0.25 the gap is +0.68 vs +0.16.

**Caveat:** this choice was made by looking at 2018-2025 outcomes. That's a small researcher
degree of freedom on the training years; the untouched 2026 holdout is the guard against it.

## 2026-10-22: Team codes normalized in the read layer

Player stats use `LV` for the Raiders in 2018-2019, while schedules, rosters, injuries, and
snap counts use `OAK`. `ml.features.load` maps OAK to LV in team columns. `game_id`s keep
`OAK` (they agree across sources) and are treated as opaque keys.

## 2026-10-23: MLflow in a separate `ml` dependency group; lockfile age floor

- **MLflow 3.16.1** (Sep 16), the final patch of the previous minor version. 3.17.0 shipped
  Oct 7 and needed two patch releases within two weeks, so it was skipped.
- **Separate group:** MLflow and its dependencies (Flask, Alembic, pandas, ...) are only for
  training and tracking. They live in a uv dependency group `ml`, synced by default for
  development. The backend Docker image installs with `--no-default-groups`, so it carries
  neither MLflow nor the dev tools.
- **`exclude-newer` in `[tool.uv]`:** the pinning policy previously covered only direct
  dependencies. Setting `exclude-newer = 2026-10-09` re-resolved about 20 *transitive* packages
  that were less than two weeks old (e.g. `certifi`, `coverage`, `httptools`, `numpy` patch
  releases) to older versions, and all checks pass. The date is a deliberate knob. Bump it when
  upgrading dependencies, then re-run CI.

## 2026-10-23: Backtest protocol

- **Walk-forward by target week.** A model predicting week W trains only on rows whose target
  week precedes W, so its labels were known by waiver time. Weekly refit by default, or per
  season to keep heavier models affordable. Random splits are never used.
- **2019-2025 evaluation, 2018 burn-in, 2026 locked:** `HOLDOUT_SEASON = 2026` in
  `ml.ingest.common`. The backtest CLI rejects it and filters its rows out on load, so the
  holdout can't be touched by accident.
- **Two scopes.** `all` rows are 44% zeros, which flatters every metric. `relevant` (top
  QB 24 / RB 48 / WR 60 / TE 24 per week by pre-week PPG, with no labels and no model output) is
  the headline scope.
- **Within-(week, position) Spearman,** averaged over weeks. Pooled correlations mostly measure
  that QBs outscore TEs.
- **Week-block bootstrap with paired differences.** Rows in a week share game conditions, so
  resampling rows would understate uncertainty. Each week reduces to sufficient statistics, so
  1,000 draws for 136k predictions take seconds.
- **Two baselines, and the stronger one is the bar.** CLAUDE.md names last-3. Season average turned
  out better on every relevant-scope position (RMSE -0.31 [-0.37, -0.26] overall). Phase 5
  must report against both; claiming a win over last-3 alone would be beating the weaker
  baseline.
- **Pickup-value metric and expert consensus deferred** to Phase 5 (FantasyPros terms check)
  and Phase 7 (optimizer).

## 2026-10-25: LightGBM and Optuna versions

- **LightGBM 4.7.0** (Jul 18) and **Optuna 4.9.0** (Jun 1), both in the `ml` group, so the
  backend image doesn't carry them. Optuna 5.0.0 (Sep 7) clears the `exclude-newer` floor, but
  it's a new major with breaking changes and no patch release yet, so the mature 4.x line was
  chosen (the same reasoning that skipped MLflow 3.17).

## 2026-10-25: Projection model design

- **One LightGBM model for all positions,** with `position` as a categorical. A per-position
  model would have a quarter of the data, and the trees can still split on position. Trained on
  every spine row with `label_ppr`, zeros included, because predicting who won't play is part
  of the task (21% of relevant player-weeks score 0).
- **Objective tuned between L2 and Tweedie.** Labels are mostly non-negative with a spike at
  zero, which Tweedie models directly. L2 is the natural choice when the metric is RMSE. The
  search picks per season. Predictions are clipped at 0.
- **Negative labels:** interceptions and fumbles lost make 0.64% of labels negative (492 of
  77,031; mean -0.59, min -6.66; mostly QBs). Tweedie rejects negative targets, so it trains on
  labels floored at 0, while L2 trains on the raw labels. Every metric is scored on the true
  labels. The first full run crashed on this, which is how it was found.
- **Fixed categorical codes** (`ml/models/matrix.py`): a hard-coded vocabulary, so a code means
  the same category in every weekly refit. Rare roster statuses (<0.1% of rows) share an
  `other` code. `last_report_status` is dropped as a duplicate of `last_report_severity`.
- **Reproducibility:** seeded bagging, `deterministic=True`, `force_row_wise=True`, and a fixed
  thread count (4), so a rerun gives identical predictions. A test checks this.
- **No early stopping on the test week.** Tree count is a tuned hyperparameter, chosen on the
  validation season.

## 2026-10-25: Nested walk-forward tuning

- For evaluation season S, Optuna (seeded TPE, 40 trials) fits on seasons before S-1 and scores
  RMSE on season S-1's `relevant` rows. The winner is then refit weekly through S. Season S
  never enters tuning, so its backtest score stays out-of-sample. Tuning on a random split, or
  once on 2019-2025, would leak.
- **Headline evaluation is 2020-2025** (101 target weeks): 2020 is the first season with two
  prior seasons of features (train on 2018, validate on 2019). Baselines are re-scored on the
  same weeks so differences stay paired.
- **Search space** (`ml/models/tune.py`): learning rate, trees, leaves, min leaf size, feature
  and bagging fractions, L2, objective, Tweedie power. It's kept small: each season's choice
  is made on one season of noisy weekly data, and a large space would overfit that season.
- Each season's params and validation RMSE are an MLflow child run under the model's run.

## 2026-10-25: Kickoff-information ablation

- The product is a Tuesday waiver tool, so its features stop at the waiver cutoff. To measure
  what that costs, `ml/features/kickoff.py` adds `ko_*` columns under a **kickoff** cutoff:
  week-W spread, total, and implied team total, plus the week-W injury designation and final
  practice level. They're tested to ignore everything after week W, and they never read
  results.
- Lines are a separate `RawData.lines` frame rather than columns on `games`, so no waiver-time
  feature can read them by accident. `ml.models.matrix.feature_columns` drops `ko_*` unless
  asked. Only the `lgbm_kickoff` model uses them, and it's reported as an ablation, not as the
  model.

## 2026-10-25: Expert consensus deferred

- FantasyPros' terms of use prohibit scraping and automated access to their rankings, and
  their historical weekly ECR archive is a paid feature. So no expert rankings are fetched or
  stored. The CLAUDE.md requirement to compare against experts is **not yet met**, and the docs
  say so instead of implying it.
- **Path to close it:** a permitted source (a licensed export, or openly published historical
  ECR) gets a loader that emits per-(player, week) rankings. The harness can score those like
  any predictor: Spearman by (week, position) needs only ranks, and the paired bootstrap applies
  unchanged.

## 2026-10-27: Screenshot parsing (Phase 6)

- **Anthropic SDK pinned at `anthropic==1.12.1`**, the newest release before the uv
  `exclude-newer` cutoff. The 2.x line (2026-10-21 onward) is too new under the pinning policy.
- **Structured outputs, not free-form JSON.** `client.messages.parse(output_format=ExtractedRoster)`
  constrains the reply to the Pydantic schema, so a successful call always yields a validated
  object. `parse()` validates inside the call, so a truncated or declined reply surfaces as a
  pydantic `ValidationError`. It's mapped to `ExtractionError` along with `refusal` and
  `max_tokens` stop reasons.
- **The model transcribes; it doesn't look players up.** The prompt says to leave team,
  position, and injury null when they aren't visible, and to mark each row `clear`, `partial`,
  or `guessed`. ID matching is done locally against nflverse rosters, so it's deterministic,
  testable, and free.
- **Model and effort come from settings** (`ANTHROPIC_MODEL`, `ANTHROPIC_EFFORT`, default
  `medium`). Effort is set explicitly because defaults differ by model. The code never names a
  model. No server-side refusal fallback is configured: the model is user-chosen, and a
  declined read is reported to the user rather than retried on another model.
- **Matching uses stdlib `difflib`**, not rapidfuzz, to avoid a dependency for about 1k
  candidates. First and last names are scored separately and averaged, so a shared surname
  ("Mike" vs "Kyle Williams") can't pass as a near-match. Abbreviated first names ("J.
  Jefferson") score 0.95 when the surname is exact. Thresholds: matched at name score >= 0.90
  with a 0.05 lead, ambiguous at >= 0.80, unmatched below. Candidates >= 0.60 are still shown.
  A team that agrees with the screenshot only breaks ties. It never lifts a weak name over a
  threshold, and a mismatch isn't penalized because screenshots lag trades. The thresholds were
  set by hand on test cases and a real-name smoke check, not fit to labeled screenshots.
- **`needs_review`** is true unless the row was read `clear`, matched uniquely, and no other
  row matched the same player. Phase 8's review screen is where those rows get corrected.
- **2026 weekly rosters are ingested for ID lookup only.** Current-season rosters are needed
  to match rookies and traded players. They're read by `backend.app.parsing.players` and never
  by features, training, or tuning, so the 2026 holdout is unaffected.
- **Images:** type comes from magic bytes (PNG, JPEG, WebP, GIF), never the filename. The size
  is capped at 5 MB (the API's per-image limit; `MAX_UPLOAD_BYTES` can only lower it). Images
  are held in memory and never written to disk.
- **Extraction accuracy is not yet measured.** It needs a labeled set of real screenshots (see
  `docs/parsing.md`). Until then, no accuracy claim is made.
