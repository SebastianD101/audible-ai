# Results

One page on what Audible AI's evidence does and doesn't show. Details, tables, and how to
reproduce each result are in the linked docs. Every number has its sample size and a 95%
interval. Weekly fantasy scoring is noisy, so differences are judged by paired intervals, not
by eye.

## The question

Can a waiver-time model of next week's PPR points make better add/drop decisions than simple
baselines? "Waiver time" means only what a manager knows on the Tuesday night after week
W-1's games: no week-W injury reports and no week-W Vegas lines.

## 1. Projections: walk-forward backtest, 2020-2025

[docs/backtest.md](backtest.md). 101 target weeks; each week is predicted by a model trained
only on earlier weeks and tuned only on earlier seasons. The headline scope is "relevant"
players: the top 156 per week by pre-week PPG, chosen without labels.

| Relevant rows (15,756 player-weeks) | RMSE | Weekly Spearman |
| --- | --- | --- |
| Season-average baseline | 9.52 | 0.265 |
| LightGBM | 7.63 [7.50, 7.75] | 0.548 [0.537, 0.560] |
| **LightGBM − season average** | **−1.89 [−2.01, −1.80]** | **+0.28 [0.27, 0.30]** |

- LightGBM beats both baselines (season average and last-3-weeks average) at every position.
- **Most of the gain is knowing who won't play.** A one-line rule (zero out players that
  pre-week data says are out) gets about half of the improvement. On players who did play,
  LightGBM's edge is real but much smaller (RMSE 8.19 → 7.92). This is a diagnostic that
  conditions on the outcome, so it has no interval.
- **Late-week information is worth about another 0.4 RMSE.** The kickoff ablation adds
  week-W lines and injury reports (not available at waiver time) and reaches 7.22. That's the
  measured cost of the waiver-time cutoff. The ablation is not the product model.

## 2. Decisions: simulated-league pickup backtest, 2020-2025

[docs/pickups.md](pickups.md). 5,050 simulated roster-weeks. The optimizer recommends the best
add/drop using each model's projections, and the move is scored on real next-week points.

| One move per week | Realized gain (points) |
| --- | --- |
| Acting on LightGBM | 1.98 [1.27, 2.75] |
| Acting on last-3 | 0.48 [−0.32, 1.37] |
| **Paired difference** | **+1.50 [0.36, 2.53]** |

- LightGBM's projected gains are calibrated (it expects +2.00 and gets +1.98). Last-3 suffers
  from the winner's curse: it expects +1.94 and gets +0.48.
- The interval is wide; 101 weeks is not a large sample. Perfect hindsight would gain about
  20 points, so there is a lot of weekly variance no model sees.
- The simulation's biases are listed in pickups.md: free agents are defined by past
  production, there are no bye-week players, and the horizon is one week.

## 3. Live 2026: the untouched holdout

[docs/live-2026.md](live-2026.md), regenerated weekly by `.github/workflows/refresh.yml`.

- 2026 is never used for tuning. Serving params are tuned on seasons up to 2025. Each week's
  projections are **committed to git before kickoff**, so the commit history is a public
  record that they couldn't have seen the results.
- Scoring uses the same metrics and bootstrap as the backtest, against the season-average
  baseline at the same cutoff. A week generated after its first kickoff is excluded
  automatically.
- **Status: tracking started at week 8. No week is scored yet.** It will take most of a season
  before the intervals mean much; until then the live page is a tracker, not a result.

## 4. Screenshot parsing and league import

[docs/parsing.md](parsing.md).

- **Extraction accuracy isn't measured.** There is no labeled screenshot set yet. The one real
  check so far is a single Sleeper screenshot, where local OCR matched 13 of 13 players. That
  is an anecdote (n = 1), not an accuracy figure.
- The parser returns per-player confidence flags, and the UI makes the user review every
  flagged row, so errors are visible rather than silent.
- **Sleeper leagues skip parsing entirely.** Exact player IDs and the league's real free-agent
  pool come from Sleeper's public API.

## What this does not show

- **Nothing about beating experts.** FantasyPros' terms don't allow using their archives, so
  there is no expert-consensus comparison.
- **PPR only.** Leagues with half-PPR, standard, TE-premium, or 6-point passing TDs get a
  warning rather than rescaled projections. Kickers, DSTs, and IDP aren't projected.
- **One-week horizon.** Rest-of-season value is approximated by the season-average hold guard.
- **Waiver status is unknown on Sleeper imports.** The public API doesn't expose it, so a
  recommended free agent may need a claim rather than an instant add.
