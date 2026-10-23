"""Which player-weeks get a projection (the spine), and what they actually scored (the labels).

Spine for target week W (W >= 2): fantasy-position players who were on a 53-man or reserve
list in week W-1 (roster status ACT, INA, or RES) or recorded a stat line earlier in the same
season. Practice-squad, cut, and retired players come in only through a stat line; including them
all would make 58% of rows trivial zeros (vs 44% now) for a 0.4-point gain in coverage of
week-W points (99.87% vs 99.48%, 2018-2025). Their team is the week W-1 roster team when
available, else the team of their latest stat line. Rows whose team has no game in week W
(bye) are dropped. Week 1 is excluded: there is no in-season data and no prior-week roster yet,
and waiver activity before week 1 is minimal.

Labels: points scored in week W, **0 if the player's team played but the player recorded no
stat line** (inactive, injured, or cut). A waiver pickup who doesn't play is worth 0, so the
target includes availability risk.

The spine reads week-W *fixtures* (who plays whom, rest days, roof). The NFL schedule is
published in May, so fixtures are known in advance; week-W *results* are not, and
:func:`fixtures` strips them.
"""

from __future__ import annotations

import polars as pl

from ml.features.asof import Week, available_before
from ml.features.load import FANTASY_POSITIONS

OUTCOME_COLUMNS = ("points_for", "points_against")
SPINE_COLUMNS = ["player_id", "name", "position", "season", "week", "team", "opponent", "game_id"]
LABEL_COLUMNS = ["target_ppr", "target_std", "target_played"]
ROSTER_STATUSES = ("ACT", "INA", "RES")


def fixtures(games: pl.DataFrame) -> pl.DataFrame:
    """Drop game outcomes from team games, leaving only what the published schedule says."""
    return games.drop(OUTCOME_COLUMNS)


def empty_spine() -> pl.DataFrame:
    """Return a zero-row spine with the standard schema."""
    return pl.DataFrame(
        schema={
            "player_id": pl.String,
            "name": pl.String,
            "position": pl.String,
            "season": pl.Int32,
            "week": pl.Int32,
            "team": pl.String,
            "opponent": pl.String,
            "game_id": pl.String,
        }
    )


def spine(
    stats: pl.DataFrame, rosters: pl.DataFrame, fixtures: pl.DataFrame, as_of: Week
) -> pl.DataFrame:
    """Return one row per candidate player for target week ``as_of``."""
    if as_of.week == 1:
        return empty_spine()
    fields = ["player_id", "name", "position", "team"]
    on_roster = (
        available_before(rosters, as_of)
        .filter(
            pl.col("season") == as_of.season,
            pl.col("week") == as_of.week - 1,
            pl.col("status").is_in(ROSTER_STATUSES),
        )
        .select(*fields, pl.lit(0).alias("priority"))
    )
    played = (
        available_before(stats, as_of)
        .filter(pl.col("season") == as_of.season, pl.col("position").is_in(FANTASY_POSITIONS))
        .sort("week")
        .group_by("player_id")
        .last()
        .select(*fields, pl.lit(1).alias("priority"))
    )
    candidates = (
        pl.concat([on_roster, played])
        .sort("priority", "player_id")
        .unique("player_id", keep="first", maintain_order=True)
        .drop("priority")
    )
    week_games = fixtures.filter(
        pl.col("season") == as_of.season, pl.col("week") == as_of.week
    ).select("season", "week", "team", "opponent", "game_id")
    return candidates.join(week_games, on="team", how="inner").select(SPINE_COLUMNS)


def add_labels(spine: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    """Attach week-W points to the spine; 0 points when the player recorded no stat line.

    This reads the target week by design. It is the only function that may, and its outputs
    are prefixed ``target_`` so they can never be mistaken for features.
    """
    scored = stats.select(
        "player_id",
        "season",
        "week",
        pl.col("fantasy_points_ppr").alias("target_ppr"),
        pl.col("fantasy_points").alias("target_std"),
    )
    return spine.join(scored, on=["player_id", "season", "week"], how="left").with_columns(
        pl.col("target_ppr").is_not_null().alias("target_played"),
        pl.col("target_ppr", "target_std").fill_null(0.0),
    )
