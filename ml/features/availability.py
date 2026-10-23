"""Availability and vacated opportunity, as known on the Tuesday before week W.

Waiver-time stand-ins for week-W injury news (which isn't published until Wednesday-Friday):

- The injury designation and practice level from the report for the team's **last game**
  before W, plus the player's roster status in week W-1.
- Whether the player missed the last team game, and how many of the last 3 team games they missed.
- Whether they **left the last game early**: snap share under half their season norm. This is
  the usual Sunday injury that drives Tuesday waiver claims.

Vacated opportunity: the season-to-date target and rush shares of teammates who are out by
the same signals (missed the last game, on reserve, or left it early). Shares freed up by an
injured starter are the classic source of waiver value.
"""

from __future__ import annotations

import polars as pl

from ml.features.asof import Week, available_before
from ml.features.usage import game_rows

LAST_N_TEAM_GAMES = 3
LEFT_EARLY_RATIO = 0.5
PRACTICE_LEVEL = {
    "Full Participation in Practice": 0,
    "Limited Participation in Practice": 1,
    "Did Not Participate In Practice": 2,
}
RESERVE_STATUSES = ("RES",)


def recent_team_games(games: pl.DataFrame, as_of: Week) -> pl.DataFrame:
    """The last ``LAST_N_TEAM_GAMES`` games each team played this season before ``as_of``."""
    return (
        available_before(games, as_of)
        .filter(pl.col("season") == as_of.season)
        .with_columns(pl.col("week").rank("ordinal", descending=True).over("team").alias("recency"))
        .filter(pl.col("recency") <= LAST_N_TEAM_GAMES)
        .select("team", "week", "game_id", "recency")
    )


def player_status(
    spine: pl.DataFrame,
    player_games: pl.DataFrame,
    team_games: pl.DataFrame,
    rosters: pl.DataFrame,
    injuries: pl.DataFrame,
    as_of: Week,
) -> pl.DataFrame:
    """Per spine player: missed games, left-early flag, last designation, roster status."""
    played = player_games.select("player_id", "game_id", "snap_pct", pl.lit(True).alias("played"))
    # Snap norm from games *other than* the team's last one, so a short game can't dilute it.
    last_game_ids = team_games.filter(pl.col("recency") == 1).select("game_id")
    norms = (
        player_games.join(last_game_ids, on="game_id", how="anti")
        .group_by("player_id")
        .agg(pl.col("snap_pct").mean().alias("snap_norm"))
    )
    per_game = (
        spine.select("player_id", "team")
        .join(team_games, on="team", how="inner")
        .join(played, on=["player_id", "game_id"], how="left")
        .with_columns(pl.col("played").fill_null(False))
    )
    last = per_game.filter(pl.col("recency") == 1).join(norms, on="player_id", how="left")
    status = last.select(
        "player_id",
        "week",
        (~pl.col("played")).alias("missed_last_game"),
        (pl.col("played") & (pl.col("snap_pct") < LEFT_EARLY_RATIO * pl.col("snap_norm")))
        .fill_null(False)
        .alias("left_last_game_early"),
    ).join(
        per_game.group_by("player_id").agg(
            (~pl.col("played")).sum().cast(pl.Int32).alias("games_missed_last3")
        ),
        on="player_id",
    )
    report = injuries.select(
        "player_id",
        "week",
        pl.col("report_status").alias("last_report_status"),
        pl.col("severity").alias("last_report_severity"),
        pl.col("practice_status")
        .replace_strict(PRACTICE_LEVEL, default=None)
        .alias("last_practice_level"),
    )
    roster = rosters.filter(pl.col("week") == as_of.week - 1).select(
        "player_id", pl.col("status").alias("roster_status_last_week")
    )
    return (
        status.join(report, on=["player_id", "week"], how="left")
        .join(roster, on="player_id", how="left")
        .drop("week")
    )


def availability_features(
    spine: pl.DataFrame,
    stats: pl.DataFrame,
    snaps: pl.DataFrame,
    rosters: pl.DataFrame,
    injuries: pl.DataFrame,
    games: pl.DataFrame,
    as_of: Week,
) -> pl.DataFrame:
    """Return availability and vacated-opportunity features keyed by ``player_id``.

    ``spine`` supplies each candidate's team and position for week ``as_of``; every other
    input is read only before ``as_of``.
    """
    this_season = pl.col("season") == as_of.season
    player_games = game_rows(
        available_before(stats, as_of).filter(this_season),
        available_before(snaps, as_of).filter(this_season),
    )
    status = player_status(
        spine,
        player_games,
        recent_team_games(games, as_of),
        available_before(rosters, as_of).filter(this_season),
        available_before(injuries, as_of).filter(this_season),
        as_of,
    )

    shares = player_games.group_by("player_id").agg(
        pl.col("target_share").mean().fill_null(0.0).alias("ts"),
        pl.col("rush_share").mean().fill_null(0.0).alias("rs"),
    )
    players = (
        spine.select("player_id", "team", "position")
        .join(status, on="player_id", how="left")
        .join(shares, on="player_id", how="left")
        .with_columns(pl.col("ts", "rs").fill_null(0.0))
    )
    out = (
        pl.col("missed_last_game").fill_null(False)
        | pl.col("left_last_game_early").fill_null(False)
        | pl.col("roster_status_last_week").is_in(RESERVE_STATUSES).fill_null(False)
    )
    freed_ts = pl.when(out).then(pl.col("ts")).otherwise(0.0)
    freed_rs = pl.when(out).then(pl.col("rs")).otherwise(0.0)
    vacated = players.with_columns(
        (freed_ts.sum().over("team", "position") - freed_ts).alias("vacated_target_share_pos"),
        (freed_ts.sum().over("team") - freed_ts).alias("vacated_target_share_team"),
        (freed_rs.sum().over("team") - freed_rs).alias("vacated_rush_share_team"),
    )
    return vacated.drop("team", "position", "ts", "rs")
