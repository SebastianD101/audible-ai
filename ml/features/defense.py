"""Defense vs position, adjusted for strength of schedule (SOS).

Raw "fantasy points allowed to WRs" punishes a defense for facing good WR rooms. Instead, for
every game a defense played before the target week, compare what the opposing offense scored at
each position with that offense's average at the position in its *other* games before the
target week:

    residual(g) = points by offense O at position P in game g
                  - mean points by O at P in O's other games this season before W

A defense's in-season rating at P is the mean residual over its games (positive = allows more
than opponents usually score). It is then shrunk toward a regressed prior-season rating:

    dvp_adj = (n * mean_residual + K * RHO * prior_season_rating) / (n + K)

so early-season ratings lean on last year, and a missing prior counts as 0 (league average).
K and RHO are fixed, documented choices, not tuned on outcomes.
"""

from __future__ import annotations

import polars as pl

from ml.features.asof import Week, available_before
from ml.features.load import FANTASY_POSITIONS

SHRINK_GAMES = 4  # K: games of evidence equal in weight to the prior.
PRIOR_SEASON_WEIGHT = 0.5  # RHO: how much of last season's rating carries over.
KEYS = ["opponent", "position"]


def position_production(stats: pl.DataFrame) -> pl.DataFrame:
    """Points per (game, offense, defense, position), with 0 where a position had no stat line."""
    fantasy = stats.filter(pl.col("position").is_in(FANTASY_POSITIONS))
    team_games = fantasy.select("season", "week", "game_id", "team", "opponent").unique()
    grid = team_games.join(pl.DataFrame({"position": list(FANTASY_POSITIONS)}), how="cross")
    points = fantasy.group_by("game_id", "team", "position").agg(
        pl.col("fantasy_points_ppr").sum().alias("points")
    )
    return grid.join(points, on=["game_id", "team", "position"], how="left").with_columns(
        pl.col("points").fill_null(0.0)
    )


def season_ratings(production: pl.DataFrame) -> pl.DataFrame:
    """Mean SOS residual and game count per (defense, position) within one season's games."""
    offense = pl.col("team"), pl.col("position")
    n_other = pl.len().over(*offense) - 1
    baseline = (pl.col("points").sum().over(*offense) - pl.col("points")) / n_other
    residuals = production.with_columns(
        pl.when(n_other > 0).then(pl.col("points") - baseline).alias("residual")
    )
    return residuals.group_by(KEYS).agg(
        pl.col("residual").mean().alias("rating"),
        pl.col("residual").count().cast(pl.Int32).alias("games"),
        pl.col("points").mean().alias("allowed"),
    )


def defense_features(stats: pl.DataFrame, as_of: Week) -> pl.DataFrame:
    """Return SOS-adjusted defense-vs-position features keyed by (opponent, position)."""
    production = position_production(available_before(stats, as_of))
    current = season_ratings(production.filter(pl.col("season") == as_of.season)).select(
        *KEYS,
        pl.col("rating").alias("dvp_season"),
        pl.col("games").alias("dvp_games"),
        pl.col("allowed").alias("dvp_raw_allowed"),
    )
    prior = season_ratings(production.filter(pl.col("season") == as_of.season - 1)).select(
        *KEYS, pl.col("rating").alias("dvp_prev")
    )
    n = pl.col("dvp_games").fill_null(0)
    evidence = n * pl.col("dvp_season").fill_null(0.0)
    prior_term = SHRINK_GAMES * PRIOR_SEASON_WEIGHT * pl.col("dvp_prev").fill_null(0.0)
    return (
        current.join(prior, on=KEYS, how="full", coalesce=True)
        .with_columns(((evidence + prior_term) / (n + SHRINK_GAMES)).alias("dvp_adj"))
        .with_columns(pl.col("dvp_games").fill_null(0))
    )
