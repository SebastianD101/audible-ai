"""Game context for the target week: venue, rest, and a scoring-environment stand-in.

Week-W venue, rest days, division flag, and roof come from the published schedule (known in
advance). The closing Vegas total is *not* known at waiver time, so the implied team total is
replaced by a stand-in built from results before week W:

    implied_total_proxy = (team points scored per game + opponent points allowed per game) / 2

Each per-game rate is shrunk toward the team's prior-season rate (league average if none) with
K games of weight, so week-2 values don't rest on a single game.
"""

from __future__ import annotations

import polars as pl

from ml.features.asof import Week, available_before

SHRINK_GAMES = 4
DEFAULT_POINTS = 22.0  # Used only if no past games exist at all.
INDOOR_ROOFS = ("dome", "closed")


def blended(season_mean: str, games: str, prior: str) -> pl.Expr:
    """Shrink an in-season per-game mean toward a prior with ``SHRINK_GAMES`` of weight."""
    n = pl.col(games).fill_null(0)
    return (n * pl.col(season_mean).fill_null(0.0) + SHRINK_GAMES * pl.col(prior)) / (
        n + SHRINK_GAMES
    )


def team_rates(games: pl.DataFrame, as_of: Week) -> pl.DataFrame:
    """Blended points-for and points-against per game, per team, from games before ``as_of``."""
    past = available_before(games, as_of).filter(pl.col("points_for").is_not_null())
    recent = past.filter(pl.col("season") >= as_of.season - 1)
    league = recent["points_for"].mean() if recent.height else None
    league_avg = float(league) if isinstance(league, int | float) else DEFAULT_POINTS

    current = (
        past.filter(pl.col("season") == as_of.season)
        .group_by("team")
        .agg(
            pl.len().alias("n"),
            pl.col("points_for").mean().alias("pf"),
            pl.col("points_against").mean().alias("pa"),
        )
    )
    prior = (
        past.filter(pl.col("season") == as_of.season - 1)
        .group_by("team")
        .agg(
            pl.col("points_for").mean().alias("prev_pf"),
            pl.col("points_against").mean().alias("prev_pa"),
        )
    )
    return (
        current.join(prior, on="team", how="full", coalesce=True)
        .with_columns(pl.col("prev_pf", "prev_pa").fill_null(league_avg))
        .select(
            "team",
            blended("pf", "n", "prev_pf").alias("points_for_pg"),
            blended("pa", "n", "prev_pa").alias("points_against_pg"),
        )
    )


def context_features(fixtures: pl.DataFrame, games: pl.DataFrame, as_of: Week) -> pl.DataFrame:
    """Return game-context features keyed by ``team`` for target week ``as_of``.

    ``fixtures`` is the published schedule (no results); ``games`` carries results and is only
    read before ``as_of``.
    """
    week = fixtures.filter(pl.col("season") == as_of.season, pl.col("week") == as_of.week)
    opp_rest = week.select(pl.col("team").alias("opponent"), pl.col("rest").alias("opp_rest"))
    rates = team_rates(games, as_of)
    opp_rates = rates.select(
        pl.col("team").alias("opponent"),
        pl.col("points_for_pg").alias("opp_points_for_pg"),
        pl.col("points_against_pg").alias("opp_points_against_pg"),
    )
    return (
        week.join(opp_rest, on="opponent", how="left")
        .join(rates, on="team", how="left")
        .join(opp_rates, on="opponent", how="left")
        .select(
            "team",
            "is_home",
            "rest",
            (pl.col("rest") - pl.col("opp_rest")).alias("rest_diff"),
            (pl.col("div_game") == 1).alias("div_game"),
            pl.col("roof").is_in(INDOOR_ROOFS).alias("indoor"),
            "points_for_pg",
            "points_against_pg",
            "opp_points_for_pg",
            "opp_points_against_pg",
            ((pl.col("points_for_pg") + pl.col("opp_points_against_pg")) / 2).alias(
                "implied_total_proxy"
            ),
        )
    )
