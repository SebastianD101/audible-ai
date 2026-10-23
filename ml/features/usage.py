"""Rolling player usage and production, from games before the target week.

In-season features aggregate the player's games this season before week W: season mean, mean of
the last 3 games played, and the trend (last 3 minus season). Volume (targets, carries) and share
(target share, rush share, snap share) are the most stable predictors of fantasy points; raw
points are noisier. Prior-season aggregates cover early weeks and players with few games.

Rows are games *played*: missed games are not zeros here (availability has its own features).
"""

from __future__ import annotations

import polars as pl

from ml.features.asof import Week, available_before

# Per-game quantities averaged in-season (season mean and last 3).
ROLLING = [
    "ppr",
    "targets",
    "receptions",
    "carries",
    "target_share",
    "air_yards_share",
    "wopr",
    "rush_share",
    "snap_pct",
    "attempts",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
]
TRENDS = ["ppr", "targets", "carries", "snap_pct"]
PRIOR_SEASON = ["ppr", "target_share", "rush_share", "snap_pct"]
LAST_N = 3


def game_rows(stats: pl.DataFrame, snaps: pl.DataFrame) -> pl.DataFrame:
    """Per player-game rows with derived shares: rush share of team carries and snap share."""
    team_carries = pl.col("carries").sum().over("game_id", "team")
    snap_pct = snaps.select("player_id", "game_id", pl.col("offense_pct").alias("snap_pct"))
    return stats.with_columns(
        pl.col("fantasy_points_ppr").alias("ppr"),
        pl.when(team_carries > 0).then(pl.col("carries") / team_carries).alias("rush_share"),
    ).join(snap_pct, on=["player_id", "game_id"], how="left")


def usage_features(stats: pl.DataFrame, snaps: pl.DataFrame, as_of: Week) -> pl.DataFrame:
    """Return usage features keyed by ``player_id`` for target week ``as_of``."""
    games = game_rows(available_before(stats, as_of), available_before(snaps, as_of))

    current = games.filter(pl.col("season") == as_of.season).with_columns(
        pl.col("week").rank("ordinal", descending=True).over("player_id").alias("recency")
    )
    recent = pl.col("recency") <= LAST_N
    in_season = current.group_by("player_id").agg(
        pl.len().cast(pl.Int32).alias("games_played"),
        (pl.lit(as_of.week) - pl.col("week").max()).cast(pl.Int32).alias("weeks_since_last_game"),
        pl.col("ppr").std().alias("ppr_season_std"),
        *(pl.col(c).mean().alias(f"{c}_season") for c in ROLLING),
        *(pl.col(c).filter(recent).mean().alias(f"{c}_last3") for c in ROLLING),
    )
    in_season = in_season.with_columns(
        (pl.col(f"{c}_last3") - pl.col(f"{c}_season")).alias(f"{c}_trend") for c in TRENDS
    )

    prior = (
        games.filter(pl.col("season") == as_of.season - 1)
        .group_by("player_id")
        .agg(
            pl.len().cast(pl.Int32).alias("prev_games"),
            *(pl.col(c).mean().alias(f"prev_{c}") for c in PRIOR_SEASON),
        )
    )
    return in_season.join(prior, on="player_id", how="full", coalesce=True).with_columns(
        pl.col("games_played").fill_null(0)
    )
