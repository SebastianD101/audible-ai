"""Download nflverse schedules (with Vegas lines) and store one Parquet file per season.

Usage:
    uv run python -m ml.ingest.schedules --start-season 2018 --end-season 2025 [--force]

Output: ``ml/data/raw/schedules/season=YYYY.parquet`` plus ``manifest.json``.

One row per game (``game_id``), regular season and postseason (``game_type`` REG/WC/DIV/CON/SB).
Besides kickoff time, rest days, roof, and division flag, each row carries **closing** Vegas
lines from nflverse:

- ``spread_line``: points the *home* team is favored by (positive = home favorite; this is the
  opposite sign of the usual sportsbook "-3.5" notation).
- ``total_line``: over/under on combined points.
- ``home_moneyline`` / ``away_moneyline``: American odds.

Leakage warning: rows also contain outcomes (``home_score``, ``away_score``, ``result``,
``total``, ``overtime``). They are kept in the raw layer for targets and evaluation only and
must never feed a feature for the same game.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy
import polars as pl

from ml.ingest import common

DEFAULT_OUT_DIR = Path("ml/data/raw/schedules")
LINE_COLUMNS = ["spread_line", "total_line"]


def fetch_schedules(season: int) -> pl.DataFrame:
    """Download the schedule for a single season from nflverse."""
    df: pl.DataFrame = nflreadpy.load_schedules(seasons=[season])
    return df


def check_lines_for_played_games(df: pl.DataFrame, season: int) -> None:
    """Require spread and total lines for every game that has a final score.

    Games not yet played (null score) may lack lines; that only happens for the current season.
    """
    played = df.filter(pl.col("home_score").is_not_null())
    missing = played.filter(pl.any_horizontal(pl.col(LINE_COLUMNS).is_null()))
    if missing.height:
        games = missing["game_id"].head(5).to_list()
        raise common.DataValidationError(
            f"schedules season {season}: {missing.height} played games missing lines, e.g. {games}"
        )


def check_one_game_per_team_week(df: pl.DataFrame, season: int) -> None:
    """Require that no team appears in more than one game in the same week."""
    teams = pl.concat(
        [
            df.select("week", pl.col("home_team").alias("team")),
            df.select("week", pl.col("away_team").alias("team")),
        ]
    )
    doubled = teams.filter(pl.len().over("week", "team") > 1).unique()
    if doubled.height:
        raise common.DataValidationError(
            f"schedules season {season}: teams with two games in one week {doubled.rows()[:5]}"
        )


SPEC = common.DatasetSpec(
    name="schedules",
    source="nflverse schedules via nflreadpy.load_schedules",
    fetch=fetch_schedules,
    default_out_dir=DEFAULT_OUT_DIR,
    required_columns={
        "game_id": pl.String,
        "season": pl.Int32,
        "game_type": pl.String,
        "week": pl.Int32,
        "gameday": pl.String,
        "gametime": pl.String,
        "away_team": pl.String,
        "home_team": pl.String,
        "away_score": pl.Int32,
        "home_score": pl.Int32,
        "away_rest": pl.Int32,
        "home_rest": pl.Int32,
        "spread_line": pl.Float64,
        "total_line": pl.Float64,
        "away_moneyline": pl.Int32,
        "home_moneyline": pl.Int32,
        "div_game": pl.Int32,
        "roof": pl.String,
    },
    key_columns=["game_id"],
    id_column="game_id",
    season_type_column="game_type",
    season_types=frozenset({"REG", "WC", "DIV", "CON", "SB"}),
    checks=(check_lines_for_played_games, check_one_game_per_team_week),
)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.run_cli(SPEC, __doc__, argv)


if __name__ == "__main__":
    main()
