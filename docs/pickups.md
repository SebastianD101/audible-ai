# Pickup backtest

How many real points do the optimizer's add/drop recommendations gain, depending on which
projections drive them? Code: `ml/backtest/pickups.py`; optimizer: `ml/optimize/`; design
choices: `docs/decisions.md` (2026-10-27).

Run `pickups-20261027T210540Z-1dd7e08`:
- predictions from walk-forward run `20261025T235213Z-6837fca`
- 2020-2025, 101 weeks, 50 simulated rosters per week (5,050 roster-weeks)
- 1,000 week-block bootstrap draws, seed 20261023
- logged to MLflow as run `a3930fd5`

```bash
uv run python -m ml.backtest.pickups --start-season 2020 --end-season 2025
```

## Method (short)

- **League:** the 156 (12 × 13) QB/RB/WR/TE with the highest `season_avg` prediction count as
  rostered. Everyone else with a game that week is a free agent.
- **Your roster:** a seeded random draw from the rostered pool: 2 QB, 4 RB, 5 WR, 2 TE.
- **Lineup:** QB, 2 RB, 2 WR, TE, FLEX. Moves need at least 0.5 projected points of gain, and
  nobody who would start on `season_avg` can be dropped.
- **Sources:** `lgbm` and `last3` projections; `oracle` uses actual points, as an upper bound.
- **Realized** means the actual PPR points of the lineup the source would start.
  - `realized_gain` is that lineup after the moves minus before, under the same source. It's
    0 when no move is made, and it's averaged over every roster-week.
  - `realized_after` is the lineup points after acting. Across sources it also measures
    knowing who to start, not just who to add.
- Intervals are 95% week-block bootstrap. The differences are paired (same rosters, same draws).

## Results

Points per roster-week, mean [95% CI].

**One move allowed**

| | lgbm | last3 | lgbm − last3 | oracle |
| --- | --- | --- | --- | --- |
| Realized gain from moves | **1.98** [1.27, 2.75] | 0.48 [−0.32, 1.37] | **+1.50** [0.36, 2.53] | 19.83 |
| Projected gain | 2.00 [1.83, 2.19] | 1.94 [1.57, 2.34] | | |
| Lineup points after acting | 95.49 | 83.48 | **+12.01** [10.63, 13.36] | 131.21 |
| Share of rosters given a move | 0.60 | 0.53 | | 1.00 |
| Moves that gained points | 0.61 [0.57, 0.66] | 0.53 [0.47, 0.58] | | 1.00 |

**Two moves allowed:** LightGBM realized gain is 2.46 [1.54, 3.46]; last3 is 0.26
[−0.94, 1.52]. The difference is +2.19 [0.79, 3.56].

## Reading it

- **LightGBM's moves gain real points; last-3's can't be told apart from zero.** About +2
  points per roster-week from one move, with an interval that excludes zero. The paired lead
  over last-3 is +1.5 [0.4, 2.5] points. The interval is wide: weekly scoring is noisy, and
  101 weeks is not a big sample.
- **LightGBM is calibrated on the gains it acts on; last-3 is not.** LightGBM expects +2.00 and
  gets +1.98. Last-3 expects +1.94 and gets +0.48. Choosing the best-looking swap selects for
  over-projection (the winner's curse), and last-3 suffers from it far more.
- **Most of the end-to-end gap is lineup-setting, not pickups.** The +12 points in lineup
  points after acting mostly comes from LightGBM knowing who won't play, as in Phase 5. Only
  about +1.5 of it is from the moves.
- **There's a lot of room left.** Perfect hindsight would gain about 20 points from one move.
  LightGBM captures roughly a tenth of that. Most of the gap is weekly variance that no model
  can see.
- **A second move adds little:** +0.5 points for LightGBM.

## Biases and limits

- **No expert comparison.** FantasyPros archives aren't accessible under their terms (see
  `docs/decisions.md`), so nothing here says anything about beating experts.
- **The ownership proxy is past production.** Free agents are, by construction, players whose
  season average looks weak. That's realistic, but real leagues also own hyped rookies and
  handcuffs, and a model that sees past box-score history has more to find in this pool than
  in a real one. `last3` overlaps with the proxy more than `lgbm` does.
- **Free agents include everyone outside the top 156.** Real waiver wires are thinner in deep
  leagues and thicker in shallow ones.
- **No bye-week players.** The spine drops teams on bye, so simulated rosters never hold them.
  Byes are a major reason managers pick players up, so this likely understates pickup value.
- **One-week horizon.** Moves are judged on next week's points only. A pickup that pays off
  later scores 0 here, and the `season_avg` hold guard is a stand-in for rest-of-season value.
- **Simulated rosters are random draws**, not real managers' teams, and K/DST are out of scope.
