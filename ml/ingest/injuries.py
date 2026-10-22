"""Download nflverse injury reports and store one Parquet file per season.

Usage:
    uv run python -m ml.ingest.injuries --start-season 2018 --end-season 2025 [--force]

Output: ``ml/data/raw/injuries/season=YYYY.parquet`` plus ``manifest.json``.

One row per player per team-week on the official injury report: game designation
(``report_status``: Out, Doubtful, Questionable, occasionally Note, or null) and practice
participation (``practice_status``).

Source quirks handled or recorded here:

- 2018-2020 deliver ``season``/``week`` as floats; they are cast to Int32 (lossless, checked)
  so every season shares one schema.
- 2025 adds ``season_type`` and drops ``date_modified``. Extra columns are kept as delivered.
- A few rows duplicate ``(gsis_id, week, team)`` where the designation changed during the week
  (e.g. Questionable then Out, 2024 week 15). Both rows are kept and counted in the manifest;
  the feature layer decides how to resolve them.
- ``practice_status`` sometimes holds whitespace instead of null; left as delivered.

Timing: game designations come from the final pre-game report, usually Friday, so they are
known before kickoff but after typical Tuesday/Wednesday waiver processing.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy
import polars as pl

from ml.ingest import common

DEFAULT_OUT_DIR = Path("ml/data/raw/injuries")
REPORT_STATUSES = frozenset({"Out", "Doubtful", "Questionable", "Note"})


def fetch_injuries(season: int) -> pl.DataFrame:
    """Download injury reports for a single season from nflverse."""
    df: pl.DataFrame = nflreadpy.load_injuries(seasons=[season])
    return df


def normalize_injuries(df: pl.DataFrame) -> pl.DataFrame:
    """Cast float ``season``/``week`` (2018-2020) to Int32."""
    return common.cast_integral_floats(df, ["season", "week"])


def check_report_statuses(df: pl.DataFrame, season: int) -> None:
    """Reject game designations outside the known set, to catch upstream changes."""
    unknown = set(df["report_status"].drop_nulls().unique().to_list()) - REPORT_STATUSES
    if unknown:
        raise common.DataValidationError(
            f"injuries season {season}: unknown report_status {sorted(unknown)}"
        )


SPEC = common.DatasetSpec(
    name="injuries",
    source="nflverse injuries via nflreadpy.load_injuries",
    fetch=fetch_injuries,
    default_out_dir=DEFAULT_OUT_DIR,
    required_columns={
        "season": pl.Int32,
        "week": pl.Int32,
        "game_type": pl.String,
        "team": pl.String,
        "gsis_id": pl.String,
        "full_name": pl.String,
        "position": pl.String,
        "report_primary_injury": pl.String,
        "report_status": pl.String,
        "practice_status": pl.String,
    },
    key_columns=["gsis_id", "week", "team"],
    id_column="gsis_id",
    season_type_column="game_type",
    season_types=frozenset({"REG", "WC", "DIV", "CON", "SB"}),
    duplicate_keys="warn",
    checks=(check_report_statuses,),
    normalize=normalize_injuries,
)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.run_cli(SPEC, __doc__, argv)


if __name__ == "__main__":
    main()
