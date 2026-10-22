"""Download nflverse snap counts and store one Parquet file per season.

Usage:
    uv run python -m ml.ingest.snap_counts --start-season 2018 --end-season 2025 [--force]

Output: ``ml/data/raw/snap_counts/season=YYYY.parquet`` plus ``manifest.json``.

One row per player per game with offensive, defensive, and special-teams snaps and the share
of team snaps (``*_pct``, 0-1). Sourced from Pro Football Reference, so players are keyed by
``pfr_player_id`` rather than gsis ID; ``ml.ingest.players`` provides the crosswalk.

``offense_pct`` is checked to lie in [0, 1]. ``st_pct`` is not checked: it reaches 1.01 in
2021 from upstream rounding, and special-teams share is not a fantasy input.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy
import polars as pl

from ml.ingest import common

DEFAULT_OUT_DIR = Path("ml/data/raw/snap_counts")


def fetch_snap_counts(season: int) -> pl.DataFrame:
    """Download snap counts for a single season from nflverse."""
    df: pl.DataFrame = nflreadpy.load_snap_counts(seasons=[season])
    return df


def check_offense_pct_range(df: pl.DataFrame, season: int) -> None:
    """Require ``offense_pct`` to be a share in [0, 1]."""
    bad = df.filter(~pl.col("offense_pct").is_between(0.0, 1.0))
    if bad.height:
        raise common.DataValidationError(
            f"snap_counts season {season}: {bad.height} rows with offense_pct outside [0, 1]"
        )


SPEC = common.DatasetSpec(
    name="snap_counts",
    source="nflverse snap_counts (Pro Football Reference) via nflreadpy.load_snap_counts",
    fetch=fetch_snap_counts,
    default_out_dir=DEFAULT_OUT_DIR,
    required_columns={
        "game_id": pl.String,
        "season": pl.Int32,
        "game_type": pl.String,
        "week": pl.Int32,
        "player": pl.String,
        "pfr_player_id": pl.String,
        "position": pl.String,
        "team": pl.String,
        "opponent": pl.String,
        "offense_snaps": pl.Float64,
        "offense_pct": pl.Float64,
    },
    key_columns=["pfr_player_id", "game_id"],
    id_column="pfr_player_id",
    season_type_column="game_type",
    season_types=frozenset({"REG", "WC", "DIV", "CON", "SB"}),
    checks=(check_offense_pct_range,),
)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.run_cli(SPEC, __doc__, argv)


if __name__ == "__main__":
    main()
