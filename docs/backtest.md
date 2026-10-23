# Backtest

How every projection model is evaluated, and the current results. Reproduce with:

```bash
uv run python -m ml.features.build --start-season 2018 --end-season 2025
uv run python -m ml.backtest.run --start-season 2019 --end-season 2025
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # browse runs
```

## Protocol

- **Walk-forward, never random splits.** To predict target week W, a model trains only on rows
  whose target week is before W (their labels are final by Tuesday of week W), then predicts
  week W from features alone. Label columns are stripped before `predict`. Tests check this
  with a spy model. Refit weekly (default) or once per season.
- **Evaluation seasons 2019-2025:** 117 target weeks (weeks 2-17 or 2-18). 2018 is burn-in
  history only. **2026 is the live holdout:** the CLI rejects it and never loads its rows.
- **Features as of waiver time** (`docs/features.md`): nothing from week W except the published
  schedule.
- **Two scopes:**
  - `all`: every spine row (44% are zeros: backups and inactive players).
  - `relevant`: per week and position, the top N by pre-week PPG (QB 24, RB 48, WR 60, TE 24;
    156 players a week), chosen without labels or model output. Use `relevant` when judging
    usefulness; `all` mainly rewards spotting backups.
- **Metrics:** RMSE and MAE on PPR points; Spearman rank correlation *within each (week,
  position)*, averaged over weeks.
- **Uncertainty:** 95% intervals from a week-block bootstrap (1,000 draws, fixed seed).
  Whole weeks are resampled because rows in a week share game conditions. Comparisons are
  **paired**: both models are scored on the same resampled weeks.
- **Tracking:** every run writes `ml/data/backtest/<run_id>/` and one MLflow run per model,
  with params, all metrics with intervals, git commit, and the features-manifest SHA-256.

## Baseline results (run `20261023T041122Z-3512959`)

Baselines project PPR points from the player's own history:
- `last3`: mean over the last 3 games played (the baseline CLAUDE.md names).
- `season_avg`: season-to-date mean.

Both fall back to prior-season PPG, then 0. Neither knows about injuries, so they over-project
players who are hurt.

#### Metrics: `all` rows

| Model | Position | Weeks | Rows | RMSE | MAE | Spearman (weekly) |
| --- | --- | ---: | ---: | --- | --- | --- |
| last3 | ALL | 117 | 67,983 | 6.397 [6.329, 6.462] | 4.190 [4.145, 4.233] | 0.578 [0.570, 0.585] |
| last3 | QB | 117 | 9,669 | 8.040 [7.910, 8.171] | 5.581 [5.461, 5.700] | 0.613 [0.598, 0.626] |
| last3 | RB | 117 | 17,611 | 6.390 [6.276, 6.503] | 4.165 [4.085, 4.241] | 0.587 [0.573, 0.600] |
| last3 | WR | 117 | 26,027 | 6.499 [6.391, 6.603] | 4.291 [4.224, 4.357] | 0.580 [0.570, 0.589] |
| last3 | TE | 117 | 14,676 | 4.805 [4.709, 4.894] | 3.125 [3.074, 3.176] | 0.531 [0.520, 0.543] |
| season_avg | ALL | 117 | 67,983 | 6.242 [6.166, 6.314] | 4.127 [4.080, 4.171] | 0.584 [0.575, 0.591] |
| season_avg | QB | 117 | 9,669 | 8.011 [7.879, 8.153] | 5.581 [5.464, 5.707] | 0.611 [0.595, 0.626] |
| season_avg | RB | 117 | 17,611 | 6.217 [6.095, 6.341] | 4.096 [4.013, 4.176] | 0.591 [0.576, 0.605] |
| season_avg | WR | 117 | 26,027 | 6.301 [6.181, 6.421] | 4.224 [4.154, 4.292] | 0.590 [0.581, 0.599] |
| season_avg | TE | 117 | 14,676 | 4.629 [4.534, 4.726] | 3.032 [2.983, 3.080] | 0.544 [0.532, 0.556] |

#### Metrics: `relevant` rows

| Model | Position | Weeks | Rows | RMSE | MAE | Spearman (weekly) |
| --- | --- | ---: | ---: | --- | --- | --- |
| last3 | ALL | 117 | 18,252 | 9.882 [9.766, 9.999] | 8.019 [7.916, 8.116] | 0.247 [0.232, 0.262] |
| last3 | QB | 117 | 2,808 | 11.133 [10.909, 11.360] | 9.045 [8.811, 9.281] | 0.253 [0.218, 0.289] |
| last3 | RB | 117 | 5,616 | 9.624 [9.444, 9.809] | 7.776 [7.616, 7.924] | 0.311 [0.287, 0.339] |
| last3 | WR | 117 | 7,020 | 10.147 [9.964, 10.326] | 8.289 [8.135, 8.450] | 0.199 [0.177, 0.221] |
| last3 | TE | 117 | 2,808 | 8.255 [8.022, 8.482] | 6.807 [6.628, 6.983] | 0.226 [0.193, 0.261] |
| season_avg | ALL | 117 | 18,252 | 9.569 [9.440, 9.704] | 7.791 [7.684, 7.905] | 0.264 [0.251, 0.277] |
| season_avg | QB | 117 | 2,808 | 11.011 [10.780, 11.251] | 8.930 [8.681, 9.175] | 0.263 [0.232, 0.296] |
| season_avg | RB | 117 | 5,616 | 9.301 [9.116, 9.495] | 7.579 [7.418, 7.734] | 0.322 [0.299, 0.347] |
| season_avg | WR | 117 | 7,020 | 9.770 [9.594, 9.975] | 8.022 [7.874, 8.184] | 0.234 [0.211, 0.255] |
| season_avg | TE | 117 | 2,808 | 7.892 [7.666, 8.116] | 6.503 [6.326, 6.675] | 0.235 [0.205, 0.269] |

#### Difference vs `last3`: `all` rows

| Model | Position | Weeks | Rows | RMSE | MAE | Spearman (weekly) |
| --- | --- | ---: | ---: | --- | --- | --- |
| season_avg | ALL | 117 | 67,983 | -0.155 [-0.180, -0.129] | -0.064 [-0.078, -0.050] | 0.007 [0.004, 0.009] |
| season_avg | QB | 117 | 9,669 | -0.028 [-0.081, 0.016] | -0.001 [-0.042, 0.038] | -0.002 [-0.007, 0.004] |
| season_avg | RB | 117 | 17,611 | -0.173 [-0.213, -0.130] | -0.069 [-0.101, -0.039] | 0.004 [0.001, 0.008] |
| season_avg | WR | 117 | 26,027 | -0.198 [-0.230, -0.163] | -0.068 [-0.088, -0.047] | 0.011 [0.008, 0.014] |
| season_avg | TE | 117 | 14,676 | -0.176 [-0.217, -0.137] | -0.093 [-0.114, -0.070] | 0.013 [0.008, 0.017] |

#### Difference vs `last3`: `relevant` rows

| Model | Position | Weeks | Rows | RMSE | MAE | Spearman (weekly) |
| --- | --- | ---: | ---: | --- | --- | --- |
| season_avg | ALL | 117 | 18,252 | -0.313 [-0.367, -0.258] | -0.228 [-0.272, -0.184] | 0.016 [0.003, 0.028] |
| season_avg | QB | 117 | 2,808 | -0.122 [-0.248, -0.020] | -0.115 [-0.237, -0.010] | 0.011 [-0.018, 0.038] |
| season_avg | RB | 117 | 5,616 | -0.322 [-0.401, -0.242] | -0.197 [-0.273, -0.119] | 0.011 [-0.005, 0.026] |
| season_avg | WR | 117 | 7,020 | -0.377 [-0.446, -0.307] | -0.267 [-0.333, -0.206] | 0.035 [0.021, 0.049] |
| season_avg | TE | 117 | 2,808 | -0.363 [-0.468, -0.267] | -0.304 [-0.398, -0.216] | 0.009 [-0.016, 0.036] |

Intervals: 95% week-block bootstrap. Diffs are model minus reference on the same resampled weeks.

### Reading these numbers

- **`season_avg` beats `last3`** on RMSE and MAE at every position, with intervals excluding 0,
  except QB on `all` rows. On `relevant` rows its RMSE is 0.31 lower [0.26, 0.37]. Recent form
  over 3 games is mostly noise relative to a full-season mean. **Any model must be compared with
  `season_avg`, not just `last3`**, or it is beating the weaker baseline.
- **Ranking startable players is hard.** Weekly Spearman is about 0.58 on all rows but 0.25-0.26
  on `relevant` rows. Most of the "all" correlation comes from separating starters from
  backups; ordering the top 156 players each week is the real test.
- **Week-to-week fantasy scoring is noisy, and the baselines ignore availability.** Relevant
  players average 10.35 PPR (QB 13.96, RB 9.98, WR 10.24, TE 7.79) with a standard deviation
  of 9.26, and **21% of relevant player-weeks score 0** (the player sat or got hurt). The baselines'
  RMSE of 9.6-9.9 is *above* that standard deviation: on RMSE alone, they're worse than
  predicting one constant for every relevant player. They still rank players better than
  chance (Spearman about 0.25), but projecting a healthy-player average for players who end up
  sitting is the biggest error to fix. The Phase 3 availability features target exactly that.
  Expect real model gains of tenths of a point to a point, judged by the paired intervals,
  not by eye.
- These are 2019-2025 walk-forward results only; nothing here has touched 2026.

## Not yet measured

- **Expert consensus** (FantasyPros): Phase 5, after checking the source's terms of use.
- **Points gained from recommended pickups vs expert pickups:** needs expert rankings and the
  Phase 7 optimizer. It will be added to this harness as another metric on the same walk-forward
  predictions.
