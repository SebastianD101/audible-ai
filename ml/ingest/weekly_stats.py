"""Download nflverse weekly player stats and store one Parquet file per season.

Usage:
    uv run python -m ml.ingest.weekly_stats --start-season 2018 --end-season 2025 [--force]

Output layout (gitignored, reproducible by re-running this module)::

    ml/data/raw/player_stats/season=2018.parquet
    ...
    ml/data/raw/player_stats/manifest.json

This is the raw layer: rows are stored as delivered by nflverse (regular season and
postseason, tagged by ``season_type``). Filtering and feature logic live in ``ml.features``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import nflreadpy
import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("ml/data/raw/player_stats")
DEFAULT_START_SEASON = 2018
# 2026 is the live holdout season (see CLAUDE.md); it is never fetched by default.
DEFAULT_END_SEASON = 2025
FIRST_AVAILABLE_SEASON = 1999

REQUIRED_COLUMNS: dict[str, type[pl.DataType]] = {
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
}
SEASON_TYPES = frozenset({"REG", "POST"})
KEY_COLUMNS = ["player_id", "season", "week", "season_type"]
# Wild card, divisional, conference championship, and Super Bowl.
POSTSEASON_WEEKS = 4


class DataValidationError(ValueError):
    """Raised when downloaded data does not match the expected shape."""


@dataclass(frozen=True)
class SeasonRecord:
    """Manifest entry describing one stored season file."""

    season: int
    file: str
    rows: int
    null_player_id_rows: int
    sha256: str


def regular_season_weeks(season: int) -> int:
    """Return the number of regular-season weeks (17 through 2020, 18 from 2021)."""
    return 17 if season <= 2020 else 18


def fetch_weekly_stats(season: int) -> pl.DataFrame:
    """Download week-level player stats for a single season from nflverse."""
    df: pl.DataFrame = nflreadpy.load_player_stats(seasons=[season], summary_level="week")
    return df


def validate_weekly_stats(df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Check schema, season, week range, and key uniqueness; return ``df`` unchanged.

    Rows with a null ``player_id`` (unattributed plays, which score zero fantasy points)
    are allowed in the raw layer and excluded from the uniqueness check.

    Raises:
        DataValidationError: If any check fails.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(f"season {season}: missing columns {missing}")

    wrong_types = {
        col: str(df.schema[col])
        for col, dtype in REQUIRED_COLUMNS.items()
        if not isinstance(df.schema[col], dtype)
    }
    if wrong_types:
        raise DataValidationError(f"season {season}: unexpected column types {wrong_types}")

    if df.is_empty():
        raise DataValidationError(f"season {season}: no rows returned")

    seasons_found = set(df["season"].unique().to_list())
    if seasons_found != {season}:
        raise DataValidationError(f"season {season}: found seasons {sorted(seasons_found)}")

    bad_types = set(df["season_type"].unique().to_list()) - SEASON_TYPES
    if bad_types:
        raise DataValidationError(f"season {season}: unexpected season_type {sorted(bad_types)}")

    reg_weeks = regular_season_weeks(season)
    out_of_range = df.filter(
        ((pl.col("season_type") == "REG") & ~pl.col("week").is_between(1, reg_weeks))
        | (
            (pl.col("season_type") == "POST")
            & ~pl.col("week").is_between(reg_weeks + 1, reg_weeks + POSTSEASON_WEEKS)
        )
    )
    if out_of_range.height:
        weeks = sorted(out_of_range["week"].unique().to_list())
        raise DataValidationError(f"season {season}: weeks out of range {weeks}")

    keyed = df.filter(pl.col("player_id").is_not_null()).select(KEY_COLUMNS)
    n_dupes = keyed.height - keyed.unique().height
    if n_dupes:
        raise DataValidationError(f"season {season}: {n_dupes} duplicate rows on {KEY_COLUMNS}")

    return df


def season_path(out_dir: Path, season: int) -> Path:
    """Return the Parquet path for a season."""
    return out_dir / f"season={season}.parquet"


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_season(df: pl.DataFrame, out_dir: Path, season: int) -> Path:
    """Write one season to Parquet (zstd), atomically via a temp file, and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = season_path(out_dir, season)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return path


def describe_season(path: Path, season: int) -> SeasonRecord:
    """Build a manifest record from a stored season file."""
    df = pl.read_parquet(path, columns=["player_id"])
    return SeasonRecord(
        season=season,
        file=path.name,
        rows=df.height,
        null_player_id_rows=df["player_id"].null_count(),
        sha256=sha256_file(path),
    )


def write_manifest(out_dir: Path, records: list[SeasonRecord]) -> Path:
    """Write ``manifest.json`` describing every stored season (used as the data version)."""
    manifest = {
        "source": "nflverse player_stats via nflreadpy.load_player_stats(summary_level='week')",
        "nflreadpy_version": version("nflreadpy"),
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seasons": [asdict(r) for r in sorted(records, key=lambda r: r.season)],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def ingest_seasons(
    seasons: list[int], out_dir: Path = DEFAULT_OUT_DIR, *, force: bool = False
) -> list[SeasonRecord]:
    """Fetch, validate, and store each season, skipping existing files unless ``force``."""
    records: list[SeasonRecord] = []
    for season in seasons:
        path = season_path(out_dir, season)
        if path.exists() and not force:
            logger.info("season %d: %s exists, skipping (use --force to refresh)", season, path)
        else:
            logger.info("season %d: downloading", season)
            df = validate_weekly_stats(fetch_weekly_stats(season), season)
            write_season(df, out_dir, season)
        record = describe_season(path, season)
        if record.null_player_id_rows:
            logger.warning(
                "season %d: %d rows with null player_id kept in raw layer",
                season,
                record.null_player_id_rows,
            )
        logger.info("season %d: %d rows in %s", season, record.rows, path)
        records.append(record)
    write_manifest(out_dir, records + stored_records(out_dir, exclude=set(seasons)))
    return records


def stored_records(out_dir: Path, exclude: set[int]) -> list[SeasonRecord]:
    """Describe season files already on disk, so a partial run keeps the manifest complete."""
    records = []
    for path in out_dir.glob("season=*.parquet"):
        season = int(path.stem.removeprefix("season="))
        if season not in exclude:
            records.append(describe_season(path, season))
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=DEFAULT_END_SEASON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="re-download existing seasons")
    args = parser.parse_args(argv)
    if args.start_season < FIRST_AVAILABLE_SEASON or args.start_season > args.end_season:
        parser.error(
            f"need {FIRST_AVAILABLE_SEASON} <= --start-season <= --end-season, "
            f"got {args.start_season}..{args.end_season}"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    seasons = list(range(args.start_season, args.end_season + 1))
    ingest_seasons(seasons, args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
