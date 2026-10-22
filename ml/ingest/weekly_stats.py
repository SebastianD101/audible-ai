"""Download nflverse weekly player stats and store one Parquet file per season.

Usage:
    uv run python -m ml.ingest.weekly_stats --start-season 2018 --end-season 2025 [--force]

Output layout (gitignored, reproducible by re-running this module)::

    ml/data/raw/player_stats/season=2018.parquet
    ...
    ml/data/raw/player_stats/manifest.json

Regular season and postseason rows are both stored, tagged by ``season_type`` (REG/POST).
About 21-22 rows per season have a null ``player_id`` (unattributed stat lines worth 0 points);
they are kept and counted in the manifest.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy
import polars as pl

from ml.ingest import common

DEFAULT_OUT_DIR = Path("ml/data/raw/player_stats")


def fetch_weekly_stats(season: int) -> pl.DataFrame:
    """Download week-level player stats for a single season from nflverse."""
    df: pl.DataFrame = nflreadpy.load_player_stats(seasons=[season], summary_level="week")
    return df


SPEC = common.DatasetSpec(
    name="player_stats",
    source="nflverse player_stats via nflreadpy.load_player_stats(summary_level='week')",
    fetch=fetch_weekly_stats,
    default_out_dir=DEFAULT_OUT_DIR,
    required_columns={
        "player_id": pl.String,
        "player_display_name": pl.String,
        "position": pl.String,
        "season": pl.Int32,
        "week": pl.Int32,
        "season_type": pl.String,
        "game_id": pl.String,
        "team": pl.String,
        "opponent_team": pl.String,
        "fantasy_points": pl.Float64,
        "fantasy_points_ppr": pl.Float64,
    },
    key_columns=["player_id", "season", "week", "season_type"],
    id_column="player_id",
    season_type_column="season_type",
    season_types=frozenset({"REG", "POST"}),
)


def validate_weekly_stats(df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Validate one season of weekly stats; see :func:`ml.ingest.common.validate`."""
    return common.validate(df, season, SPEC)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.run_cli(SPEC, __doc__, argv)


if __name__ == "__main__":
    main()
