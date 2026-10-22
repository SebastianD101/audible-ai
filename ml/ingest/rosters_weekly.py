"""Download nflverse weekly rosters and store one Parquet file per season.

Usage:
    uv run python -m ml.ingest.rosters_weekly --start-season 2018 --end-season 2025 [--force]

Output: ``ml/data/raw/rosters_weekly/season=YYYY.parquet`` plus ``manifest.json``.

One row per player per week (``gsis_id``, ``week``) with team, position, depth-chart position,
roster ``status`` (ACT, RES, INA, CUT, ...; the code set varies by season), experience, and
cross-platform IDs (``sleeper_id``, ``espn_id``, ``yahoo_id``, ``pfr_id``) used later to match
parsed screenshot rosters. A handful of rows per season have a null ``gsis_id``; they are kept
and counted in the manifest.

Duplicate ``(gsis_id, week)`` rows are kept and counted rather than rejected, because upstream
has at least one ID collision: in 2019 weeks 2-17, Isaiah Searight (NYG) is listed under Quinnen
Williams's gsis ID ``00-0035718`` (Searight's own ID is ``00-0035478``). Joins on these keys must
deduplicate or check the name against the ``ml.ingest.players`` crosswalk.

nflverse does not publish the time each weekly snapshot was taken, so ``status`` should be
treated as "status for that week" rather than a timestamped pre-kickoff observation.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy
import polars as pl

from ml.ingest import common

DEFAULT_OUT_DIR = Path("ml/data/raw/rosters_weekly")


def fetch_rosters_weekly(season: int) -> pl.DataFrame:
    """Download weekly rosters for a single season from nflverse."""
    df: pl.DataFrame = nflreadpy.load_rosters_weekly(seasons=[season])
    return df


SPEC = common.DatasetSpec(
    name="rosters_weekly",
    source="nflverse weekly rosters via nflreadpy.load_rosters_weekly",
    fetch=fetch_rosters_weekly,
    default_out_dir=DEFAULT_OUT_DIR,
    required_columns={
        "season": pl.Int32,
        "week": pl.Int32,
        "game_type": pl.String,
        "team": pl.String,
        "gsis_id": pl.String,
        "full_name": pl.String,
        "position": pl.String,
        "depth_chart_position": pl.String,
        "status": pl.String,
        "years_exp": pl.Int32,
        "pfr_id": pl.String,
        "sleeper_id": pl.String,
        "espn_id": pl.String,
        "yahoo_id": pl.String,
    },
    key_columns=["gsis_id", "week"],
    id_column="gsis_id",
    season_type_column="game_type",
    season_types=frozenset({"REG", "WC", "DIV", "CON", "SB"}),
    duplicate_keys="warn",
)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.run_cli(SPEC, __doc__, argv)


if __name__ == "__main__":
    main()
