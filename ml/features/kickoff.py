"""Kickoff-information features: an ablation, never inputs to the product model.

The product makes waiver recommendations on Tuesday, before week-W lines settle and before
injury reports come out, so every regular feature stops at the waiver-time cutoff
(``ml.features.asof``). These ``ko_*`` features relax the cutoff to **kickoff of week W**. That
measures how much the late-week information is worth, which tells us how much a Tuesday
projection gives up. They read only week-W lines and week-W injury reports, never results.
"""

from __future__ import annotations

import polars as pl

from ml.features.asof import Week, available_before
from ml.features.availability import PRACTICE_LEVEL


def available_through[Frame: (pl.DataFrame, pl.LazyFrame)](df: Frame, as_of: Week) -> Frame:
    """Keep rows from weeks up to and including ``as_of`` (the kickoff cutoff)."""
    return available_before(df, Week(as_of.season, as_of.week + 1))


def at_week(df: pl.DataFrame, as_of: Week) -> pl.DataFrame:
    """Rows for week ``as_of`` itself, read through the kickoff cutoff."""
    return available_through(df, as_of).filter(
        pl.col("season") == as_of.season, pl.col("week") == as_of.week
    )


def kickoff_features(
    spine: pl.DataFrame, lines: pl.DataFrame, injuries: pl.DataFrame, as_of: Week
) -> pl.DataFrame:
    """Return ``ko_*`` features keyed by ``player_id``.

    Args:
        spine: Week-``as_of`` candidates with their ``team``.
        lines: ``load.game_lines`` output.
        injuries: ``load.clean_injuries`` output.
        as_of: The target week; its own lines and reports are used, nothing later.
    """
    # One row per key even if an input repeats one, so the joins can't fan out spine rows.
    week_lines = at_week(lines, as_of).unique("team", keep="first", maintain_order=True)
    week_lines = week_lines.select(
        "team",
        pl.col("team_spread").alias("ko_team_spread"),
        pl.col("total_line").alias("ko_total_line"),
        ((pl.col("total_line") + pl.col("team_spread")) / 2).alias("ko_implied_team_total"),
    )
    week_reports = (
        at_week(injuries, as_of)
        .sort("severity", descending=True, nulls_last=True)
        .unique("player_id", keep="first", maintain_order=True)
    )
    report = week_reports.select(
        "player_id",
        pl.col("severity").alias("ko_report_severity"),
        pl.col("practice_status")
        .replace_strict(PRACTICE_LEVEL, default=None)
        .alias("ko_practice_level"),
    )
    return (
        spine.select("player_id", "team")
        .join(week_lines, on="team", how="left")
        .join(report, on="player_id", how="left")
        .drop("team")
    )
