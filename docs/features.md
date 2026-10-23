# Feature dictionary

One row per (player, target week W), built by `uv run python -m ml.features.build`.
2018-2025: 77,031 rows, 63 features. Week 1 and bye weeks are excluded (see `docs/decisions.md`).

**Information cutoff: waiver time.** Every feature uses only data from weeks before W, as known
on the Tuesday night after week W-1. The one exception is the **published schedule** for week W
(opponent, venue, rest, roof), which is released in May. Week-W results, closing lines, injury
designations, and practice reports are never read. Each feature function applies the cutoff
itself, and each has a test that corrupts all future rows and requires identical output.

## Identifiers (not model inputs)

`player_id` (gsis), `name`, `position` (QB/RB/WR/TE), `season`, `week` (target week W),
`team`, `opponent`, `game_id`.

## Labels (not model inputs)

| Column | Definition |
| --- | --- |
| `label_ppr` | PPR fantasy points in week W; **0 if the team played and the player had no stat line** |
| `label_std` | Standard-scoring points, same rule |
| `label_played` | Whether the player recorded a stat line in week W |

Labels are prefixed `label_`, not `target_`, so they can't be confused with `target_share`.

## Usage (`ml/features/usage.py`)

Over games the player **played** this season before W (missed games aren't zeros here).

| Feature | Definition |
| --- | --- |
| `games_played` | Games with a stat line this season before W |
| `weeks_since_last_game` | W minus the week of the player's last game |
| `{stat}_season` | Mean per game this season |
| `{stat}_last3` | Mean over the last 3 games played |
| `{ppr,targets,carries,snap_pct}_trend` | `_last3` minus `_season` |
| `ppr_season_std` | Std of PPR points per game this season |
| `prev_games`, `prev_{ppr,target_share,rush_share,snap_pct}` | Prior full season |

`{stat}` is one of `ppr`, `targets`, `receptions`, `carries`, `target_share`, `air_yards_share`,
`wopr`, `rush_share` (player carries / team carries in the game), `snap_pct` (offensive snap
share, from PFR via the ID crosswalk), `attempts`, `passing_yards`, `rushing_yards`,
`receiving_yards`.

## Defense vs position, SOS-adjusted (`ml/features/defense.py`)

Keyed by the week-W opponent and the player's position.

| Feature | Definition |
| --- | --- |
| `dvp_raw_allowed` | Mean PPR points the defense allowed to the position per game this season |
| `dvp_season` | Mean SOS residual: opposing offense's points at the position minus that offense's average in its *other* games before W |
| `dvp_games` | Games behind `dvp_season` |
| `dvp_prev` | Same residual rating over the full prior season |
| `dvp_adj` | `(n * dvp_season + 4 * 0.5 * dvp_prev) / (n + 4)`: shrunk toward a regressed prior |

## Game context (`ml/features/context.py`)

| Feature | Definition | Source |
| --- | --- | --- |
| `is_home`, `rest`, `rest_diff`, `div_game`, `indoor` | Week-W venue, rest days (and vs opponent), division game, dome/closed roof | Published schedule |
| `points_for_pg`, `points_against_pg` | Team points scored/allowed per game before W, shrunk toward the prior season (K=4) | Results before W |
| `opp_points_for_pg`, `opp_points_against_pg` | Same for the opponent | Results before W |
| `implied_total_proxy` | `(points_for_pg + opp_points_against_pg) / 2`, a stand-in for the closing team total | Results before W |

## Availability and vacated opportunity (`ml/features/availability.py`)

"Last game" is the team's most recent game before W, so a bye in W-1 doesn't blank these out.

| Feature | Definition |
| --- | --- |
| `missed_last_game` | Team played its last game, player had no stat line |
| `left_last_game_early` | Played, but snap share < 50% of their norm from other games this season |
| `games_missed_last3` | Missed games among the team's last 3 |
| `last_report_status`, `last_report_severity` | Designation for the last game (Note 0, Questionable 1, Doubtful 2, Out 3; null = not listed) |
| `last_practice_level` | Final practice participation that week (Full 0, Limited 1, DNP 2) |
| `roster_status_last_week` | Roster status in week W-1 (ACT, INA, RES, ...) |
| `vacated_target_share_pos` / `_team` | Season target share of same-position / all teammates who are **newly** out |
| `vacated_rush_share_team` | Season rush share of teammates who are newly out |

"Newly out" means the teammate left the last game early, or played the game before last and
then missed the last game or went on reserve. Season-long absences aren't counted, because their
roles were redistributed long ago.

## Descriptive checks (2018-2025, not model results)

Spearman correlation with `label_ppr`, by position, for rows with at least one game this season:
recent production and usage are the strongest single signals (`ppr_last3` 0.50-0.63,
`snap_pct_last3` 0.52-0.59), and `missed_last_game` is strongly negative (-0.43 to -0.61).

Opponent and opportunity effects are small next to player quality, so they were checked against
`label_ppr - ppr_season` (players with 3+ games who played week W, n = 33,042):

- `dvp_adj`: QB 0.106, RB 0.079, TE 0.048, WR 0.031. The SOS-adjusted rating beats raw points
  allowed at every position.
- `implied_total_proxy`: about 0 at every position. Results before W are a poor substitute for a
  Vegas total. The Phase 5 kickoff-information ablation will measure what the real line adds.
- Newly vacated share: WR/TE above 0.15 averaged +0.21 PPG over their season mean (vs +0.03);
  RBs with vacated rush share above 0.25 averaged +0.68 (vs +0.16).

These are single-split descriptive numbers without confidence intervals. Model comparisons with
bootstrapped intervals come in Phases 4-5.
