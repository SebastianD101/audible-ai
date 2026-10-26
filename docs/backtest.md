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

## LightGBM results (run `20261025T235213Z-6837fca`)

Reproduce:

```bash
uv run python -m ml.backtest.run --start-season 2020 --end-season 2025 \
  --models last3,season_avg,lgbm,lgbm_kickoff
```

- **2020-2025, 101 target weeks.** 2020 is the first season where nested tuning has two prior
  seasons: train on 2018, validate on 2019. Baselines are re-scored on the same weeks, so their
  numbers differ slightly from the 2019-2025 table above.
- **`lgbm`** is the product model: waiver-time features only, tuned per season (40 Optuna
  trials on season S-1's relevant rows), refit weekly.
- **`lgbm_kickoff`** is an **ablation, not the product**. It adds week-W Vegas lines and the
  week-W injury report, which aren't known at waiver time.
- **Headline scope is `relevant`; the bar is `season_avg`.**

#### `relevant` rows

| Model | Position | Weeks | Rows | RMSE | MAE | Spearman (weekly) |
| --- | --- | ---: | ---: | --- | --- | --- |
| season_avg | ALL | 101 | 15,756 | 9.518 [9.380, 9.657] | 7.750 [7.631, 7.874] | 0.265 [0.249, 0.280] |
| season_avg | QB | 101 | 2,424 | 11.106 [10.866, 11.359] | 9.010 [8.763, 9.274] | 0.262 [0.230, 0.297] |
| season_avg | RB | 101 | 4,848 | 9.302 [9.096, 9.501] | 7.573 [7.399, 7.738] | 0.317 [0.288, 0.345] |
| season_avg | WR | 101 | 6,060 | 9.606 [9.425, 9.786] | 7.907 [7.759, 8.069] | 0.258 [0.235, 0.280] |
| season_avg | TE | 101 | 2,424 | 7.860 [7.599, 8.140] | 6.450 [6.253, 6.655] | 0.222 [0.188, 0.258] |
| lgbm | ALL | 101 | 15,756 | 7.625 [7.501, 7.749] | 5.690 [5.595, 5.780] | 0.548 [0.537, 0.560] |
| lgbm | QB | 101 | 2,424 | 8.002 [7.750, 8.263] | 6.085 [5.862, 6.321] | 0.560 [0.533, 0.587] |
| lgbm | RB | 101 | 4,848 | 7.616 [7.416, 7.818] | 5.613 [5.469, 5.765] | 0.581 [0.562, 0.600] |
| lgbm | WR | 101 | 6,060 | 7.871 [7.681, 8.065] | 5.932 [5.788, 6.068] | 0.524 [0.507, 0.544] |
| lgbm | TE | 101 | 2,424 | 6.558 [6.260, 6.896] | 4.843 [4.680, 5.025] | 0.526 [0.500, 0.552] |
| lgbm_kickoff | ALL | 101 | 15,756 | 7.216 [7.100, 7.339] | 5.302 [5.219, 5.381] | 0.618 [0.609, 0.629] |
| lgbm_kickoff | QB | 101 | 2,424 | 7.466 [7.224, 7.725] | 5.632 [5.422, 5.852] | 0.626 [0.597, 0.654] |
| lgbm_kickoff | RB | 101 | 4,848 | 7.148 [6.958, 7.346] | 5.194 [5.057, 5.326] | 0.662 [0.646, 0.678] |
| lgbm_kickoff | WR | 101 | 6,060 | 7.506 [7.312, 7.692] | 5.545 [5.413, 5.677] | 0.597 [0.578, 0.615] |
| lgbm_kickoff | TE | 101 | 2,424 | 6.298 [6.000, 6.618] | 4.576 [4.410, 4.755] | 0.588 [0.567, 0.612] |

#### Difference vs `season_avg`: `relevant` rows

| Model | Position | Weeks | Rows | RMSE | MAE | Spearman (weekly) |
| --- | --- | ---: | ---: | --- | --- | --- |
| lgbm | ALL | 101 | 15,756 | -1.893 [-2.013, -1.795] | -2.060 [-2.151, -1.979] | 0.283 [0.270, 0.297] |
| lgbm | QB | 101 | 2,424 | -3.105 [-3.377, -2.872] | -2.925 [-3.188, -2.689] | 0.298 [0.267, 0.329] |
| lgbm | RB | 101 | 4,848 | -1.686 [-1.844, -1.544] | -1.960 [-2.096, -1.832] | 0.264 [0.243, 0.288] |
| lgbm | WR | 101 | 6,060 | -1.735 [-1.904, -1.586] | -1.975 [-2.107, -1.845] | 0.267 [0.245, 0.290] |
| lgbm | TE | 101 | 2,424 | -1.302 [-1.457, -1.143] | -1.607 [-1.752, -1.459] | 0.305 [0.274, 0.335] |
| lgbm_kickoff | ALL | 101 | 15,756 | -2.303 [-2.415, -2.200] | -2.448 [-2.552, -2.358] | 0.354 [0.338, 0.370] |
| lgbm_kickoff | QB | 101 | 2,424 | -3.641 [-3.944, -3.371] | -3.377 [-3.667, -3.120] | 0.364 [0.332, 0.401] |
| lgbm_kickoff | RB | 101 | 4,848 | -2.154 [-2.334, -1.988] | -2.379 [-2.543, -2.233] | 0.344 [0.319, 0.371] |
| lgbm_kickoff | WR | 101 | 6,060 | -2.100 [-2.252, -1.953] | -2.361 [-2.494, -2.231] | 0.339 [0.315, 0.363] |
| lgbm_kickoff | TE | 101 | 2,424 | -1.562 [-1.722, -1.388] | -1.873 [-2.029, -1.718] | 0.367 [0.332, 0.400] |

#### Difference vs `season_avg`: `all` rows

| Model | Position | Weeks | Rows | RMSE | MAE | Spearman (weekly) |
| --- | --- | ---: | ---: | --- | --- | --- |
| lgbm | ALL | 101 | 58,851 | -1.140 [-1.195, -1.093] | -1.081 [-1.116, -1.049] | 0.133 [0.128, 0.139] |
| lgbm | QB | 101 | 8,334 | -2.244 [-2.380, -2.119] | -2.163 [-2.268, -2.059] | 0.124 [0.112, 0.136] |
| lgbm | RB | 101 | 15,260 | -1.005 [-1.079, -0.939] | -0.995 [-1.048, -0.947] | 0.151 [0.143, 0.160] |
| lgbm | WR | 101 | 22,533 | -0.965 [-1.043, -0.893] | -0.961 [-1.009, -0.910] | 0.128 [0.120, 0.137] |
| lgbm | TE | 101 | 12,724 | -0.666 [-0.719, -0.611] | -0.690 [-0.727, -0.653] | 0.128 [0.120, 0.137] |
| lgbm_kickoff | ALL | 101 | 58,851 | -1.352 [-1.405, -1.304] | -1.198 [-1.232, -1.166] | 0.163 [0.157, 0.169] |
| lgbm_kickoff | QB | 101 | 8,334 | -2.566 [-2.711, -2.430] | -2.321 [-2.433, -2.211] | 0.146 [0.133, 0.160] |
| lgbm_kickoff | RB | 101 | 15,260 | -1.250 [-1.327, -1.177] | -1.125 [-1.183, -1.072] | 0.188 [0.178, 0.199] |
| lgbm_kickoff | WR | 101 | 22,533 | -1.158 [-1.233, -1.086] | -1.094 [-1.142, -1.045] | 0.163 [0.155, 0.172] |
| lgbm_kickoff | TE | 101 | 12,724 | -0.774 [-0.828, -0.717] | -0.734 [-0.770, -0.697] | 0.155 [0.146, 0.164] |

Intervals: 95% week-block bootstrap, paired on the same resampled weeks. Diffs vs `last3`,
which are larger, and the `all`-scope values are in the run's `metrics.md` and in MLflow.

### Reading these numbers

- **LightGBM beats both baselines at every position, with intervals well clear of 0.** On
  relevant rows vs `season_avg`: RMSE -1.89 [-2.01, -1.80] (9.52 to 7.63), MAE -2.06, and weekly
  Spearman +0.28 [0.27, 0.30] (0.265 to 0.548). That's 101 weeks and 15,756 player-weeks.
  The gain is largest for QBs (RMSE -3.1) and smallest for TEs (-1.3).
- **Most of the gain is availability, not better forecasts of healthy players.** The baselines
  project a full-season average for players who are about to sit. These checks are diagnostics
  without intervals, and the second one conditions on the outcome, so it isn't a metric:
  - Relevant players who **did play** week W (12,648 rows): `season_avg` RMSE 8.19 and
    Spearman 0.278, vs `lgbm` 7.92 and 0.340. That's a real but much smaller gain.
  - A **one-line heuristic**, `season_avg` set to 0 when pre-week data says the player is out
    (roster status RES/INA/CUT/DEV/SUS, or missed the last game while listed Out), already
    reaches RMSE 8.55 and Spearman 0.444. That's about half of LightGBM's improvement, from
    one rule. LightGBM's edge over it comes from grading availability (doubtful returns,
    snap-share trends) and from small usage and matchup effects.
  - Top features by gain (summed over seasons): `ppr_season` 27%, `roster_status_last_week`
    19%, `ppr_last3` 11%, `missed_last_game` 8%, `snap_pct_last3` 6%,
    `weeks_since_last_game` 6%. Matchup features (`dvp_*`) are under 1% each.
- **Leakage checks on a result this large.** Every feature has a corrupt-the-future test, and
  tuning has a spy test. `roster_status_last_week` carries 19% of the gain, so its timing was
  checked directly: of 10,537 weekly snapshots with status `RES`, **none** played that week's
  game, and `INA` (game-day inactives) is present. So the week W-1 snapshot is fixed by W-1's
  game day, before Tuesday waivers. It doesn't capture post-waiver IR moves.
- **Late-week information is worth about 0.4 points of RMSE.** Paired, `lgbm_kickoff` minus
  `lgbm` on relevant rows: RMSE -0.41 [-0.46, -0.37], Spearman +0.070 [0.062, 0.079]. By
  position, RMSE drops most for QBs (-0.54 [-0.67, -0.41]) and least for TEs (-0.26
  [-0.33, -0.19]). That's what a Tuesday recommendation gives up by design. It also argues for a projection refresh after Friday
  injury reports in a later phase.
- **Still noisy.** Relevant-row RMSE of 7.6 against a mean of about 10 points means single-week
  calls stay unreliable. A Spearman of 0.55 is a meaningful ranking, not a precise one. Gains
  are averages over 101 weeks, not promises for a given week.
- **Validation RMSE during tuning** (season S-1) ran 7.3-8.2 for `lgbm`, close to the test
  seasons' 7.6, so tuning didn't overfit its validation season visibly.
- Nothing here touches 2026.

## Not yet measured

- **Expert consensus: not compared.** FantasyPros' terms of use prohibit scraping and automated
  access, and their historical weekly rankings are a paid feature. So no expert rankings were
  fetched (see `docs/decisions.md`). **The project can't yet claim anything about experts.**
  Once a permitted source exists, expert ranks plug into the same harness: weekly Spearman
  needs only ranks.
- **Points gained from recommended pickups vs expert pickups:** needs expert rankings and the
  Phase 7 optimizer.
- **2026 live holdout:** untouched. It's scored once, at the end, with the frozen model.
