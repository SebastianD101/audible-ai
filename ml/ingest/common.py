"""Shared machinery for nflverse raw-layer loaders.

Each dataset is described by a :class:`DatasetSpec`; this module fetches, validates, and stores
it as one Parquet file per season plus a ``manifest.json`` that serves as the data version::

    <out_dir>/season=2018.parquet
    ...
    <out_dir>/manifest.json

The raw layer stores rows as delivered (extra or season-specific columns are kept). Filtering
and feature logic live in ``ml.features``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_START_SEASON = 2018
# 2026 is the live holdout season (see CLAUDE.md); it is never fetched by default.
DEFAULT_END_SEASON = 2025
FIRST_AVAILABLE_SEASON = 1999
# Wild card, divisional, conference championship, and Super Bowl.
POSTSEASON_WEEKS = 4
REGULAR_SEASON = "REG"

Check = Callable[[pl.DataFrame, int], None]
Normalize = Callable[[pl.DataFrame], pl.DataFrame]


class DataValidationError(ValueError):
    """Raised when downloaded data does not match the expected shape."""


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of one per-season nflverse dataset.

    Attributes:
        name: Short dataset name used in logs and the manifest.
        source: Human-readable provenance recorded in the manifest.
        fetch: Downloads one season.
        default_out_dir: Where season files go unless ``--out-dir`` is given.
        required_columns: Columns that must exist with these dtypes. Other columns are kept
            as delivered, which absorbs season-to-season schema drift.
        key_columns: Columns that identify a row.
        id_column: Entity ID column; null IDs are allowed, counted, and excluded from the
            key-uniqueness check.
        season_type_column: Column tagging regular season vs postseason rows.
        season_types: Allowed values of ``season_type_column``; every value other than
            ``"REG"`` is treated as postseason.
        duplicate_keys: ``"error"`` rejects duplicate keys; ``"warn"`` keeps and counts them.
        checks: Extra dataset-specific checks, each raising :class:`DataValidationError`.
        normalize: Optional dtype fix-up applied before validation, for sources whose
            schema drifts between seasons. Keep it to lossless casts; cleaning belongs in
            ``ml.features``.
    """

    name: str
    source: str
    fetch: Callable[[int], pl.DataFrame]
    default_out_dir: Path
    required_columns: dict[str, type[pl.DataType]]
    key_columns: list[str]
    id_column: str
    season_type_column: str
    season_types: frozenset[str]
    duplicate_keys: Literal["error", "warn"] = "error"
    checks: tuple[Check, ...] = field(default=())
    normalize: Normalize | None = None


@dataclass(frozen=True)
class SeasonRecord:
    """Manifest entry describing one stored season file."""

    season: int
    file: str
    rows: int
    null_id_rows: int
    duplicate_key_rows: int
    sha256: str


def regular_season_weeks(season: int) -> int:
    """Return the number of regular-season weeks (17 through 2020, 18 from 2021)."""
    return 17 if season <= 2020 else 18


def cast_integral_floats(
    df: pl.DataFrame, columns: list[str], dtype: type[pl.DataType] = pl.Int32
) -> pl.DataFrame:
    """Cast float ``columns`` to ``dtype`` when every non-null value is a whole number.

    Raises:
        DataValidationError: If a float column holds a non-integral value, so the cast
            would lose information.
    """
    floats = [c for c in columns if c in df.columns and df.schema[c].is_float()]
    for col in floats:
        if (df[col].drop_nulls() % 1 != 0).any():
            raise DataValidationError(f"column {col!r} has non-integral values")
    return df.with_columns(pl.col(floats).cast(dtype)) if floats else df


def count_duplicate_keys(df: pl.DataFrame, spec: DatasetSpec) -> int:
    """Count rows beyond the first for each key, ignoring rows with a null ID."""
    keyed = df.filter(pl.col(spec.id_column).is_not_null()).select(spec.key_columns)
    return keyed.height - keyed.unique().height


def validate(df: pl.DataFrame, season: int, spec: DatasetSpec) -> pl.DataFrame:
    """Check schema, season, week range, key uniqueness, and spec checks; return ``df``.

    Raises:
        DataValidationError: If any check fails.
    """
    label = f"{spec.name} season {season}"
    missing = sorted(set(spec.required_columns) - set(df.columns))
    if missing:
        raise DataValidationError(f"{label}: missing columns {missing}")

    wrong_types = {
        col: str(df.schema[col])
        for col, dtype in spec.required_columns.items()
        if not isinstance(df.schema[col], dtype)
    }
    if wrong_types:
        raise DataValidationError(f"{label}: unexpected column types {wrong_types}")

    if df.is_empty():
        raise DataValidationError(f"{label}: no rows returned")

    seasons_found = set(df["season"].unique().to_list())
    if seasons_found != {season}:
        raise DataValidationError(f"{label}: found seasons {sorted(seasons_found)}")

    type_col = pl.col(spec.season_type_column)
    bad_types = set(df[spec.season_type_column].unique().to_list()) - spec.season_types
    if bad_types:
        raise DataValidationError(
            f"{label}: unexpected {spec.season_type_column} {sorted(bad_types, key=str)}"
        )

    reg_weeks = regular_season_weeks(season)
    out_of_range = df.filter(
        ((type_col == REGULAR_SEASON) & ~pl.col("week").is_between(1, reg_weeks))
        | (
            (type_col != REGULAR_SEASON)
            & ~pl.col("week").is_between(reg_weeks + 1, reg_weeks + POSTSEASON_WEEKS)
        )
    )
    if out_of_range.height:
        weeks = sorted(out_of_range["week"].unique().to_list())
        raise DataValidationError(f"{label}: weeks out of range {weeks}")

    n_dupes = count_duplicate_keys(df, spec)
    if n_dupes and spec.duplicate_keys == "error":
        raise DataValidationError(f"{label}: {n_dupes} duplicate rows on {spec.key_columns}")

    for check in spec.checks:
        check(df, season)

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


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> Path:
    """Write ``df`` to Parquet (zstd) via a temp file so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return path


def write_season(df: pl.DataFrame, out_dir: Path, season: int) -> Path:
    """Write one season atomically and return its path."""
    return write_parquet_atomic(df, season_path(out_dir, season))


def describe_season(path: Path, season: int, spec: DatasetSpec) -> SeasonRecord:
    """Build a manifest record from a stored season file."""
    df = pl.read_parquet(path, columns=list(dict.fromkeys([spec.id_column, *spec.key_columns])))
    return SeasonRecord(
        season=season,
        file=path.name,
        rows=df.height,
        null_id_rows=df[spec.id_column].null_count(),
        duplicate_key_rows=count_duplicate_keys(df, spec),
        sha256=sha256_file(path),
    )


def write_manifest(out_dir: Path, spec: DatasetSpec, records: list[SeasonRecord]) -> Path:
    """Write ``manifest.json`` describing every stored season (used as the data version)."""
    manifest = {
        "dataset": spec.name,
        "source": spec.source,
        "nflreadpy_version": version("nflreadpy"),
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "id_column": spec.id_column,
        "key_columns": spec.key_columns,
        "seasons": [asdict(r) for r in sorted(records, key=lambda r: r.season)],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def stored_records(out_dir: Path, spec: DatasetSpec, exclude: set[int]) -> list[SeasonRecord]:
    """Describe season files already on disk, so a partial run keeps the manifest complete."""
    records = []
    for path in out_dir.glob("season=*.parquet"):
        season = int(path.stem.removeprefix("season="))
        if season not in exclude:
            records.append(describe_season(path, season, spec))
    return records


def ingest_seasons(
    spec: DatasetSpec, seasons: list[int], out_dir: Path | None = None, *, force: bool = False
) -> list[SeasonRecord]:
    """Fetch, validate, and store each season, skipping existing files unless ``force``."""
    out_dir = out_dir or spec.default_out_dir
    records: list[SeasonRecord] = []
    for season in seasons:
        path = season_path(out_dir, season)
        if path.exists() and not force:
            logger.info(
                "%s %d: %s exists, skipping (use --force to refresh)", spec.name, season, path
            )
        else:
            logger.info("%s %d: downloading", spec.name, season)
            df = spec.fetch(season)
            if spec.normalize is not None:
                df = spec.normalize(df)
            df = validate(df, season, spec)
            write_season(df, out_dir, season)
        record = describe_season(path, season, spec)
        if record.null_id_rows:
            logger.warning(
                "%s %d: %d rows with null %s kept in raw layer",
                spec.name,
                season,
                record.null_id_rows,
                spec.id_column,
            )
        if record.duplicate_key_rows:
            logger.warning(
                "%s %d: %d duplicate rows on %s kept in raw layer",
                spec.name,
                season,
                record.duplicate_key_rows,
                spec.key_columns,
            )
        logger.info("%s %d: %d rows in %s", spec.name, season, record.rows, path)
        records.append(record)
    write_manifest(out_dir, spec, records + stored_records(out_dir, spec, exclude=set(seasons)))
    return records


def build_parser(description: str | None, default_out_dir: Path) -> argparse.ArgumentParser:
    """Return the shared ``--start-season/--end-season/--out-dir/--force`` parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=DEFAULT_END_SEASON)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--force", action="store_true", help="re-download existing data")
    return parser


def parse_season_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    """Parse arguments and reject season ranges outside what nflverse provides."""
    args = parser.parse_args(argv)
    if args.start_season < FIRST_AVAILABLE_SEASON or args.start_season > args.end_season:
        parser.error(
            f"need {FIRST_AVAILABLE_SEASON} <= --start-season <= --end-season, "
            f"got {args.start_season}..{args.end_season}"
        )
    return args


def configure_logging() -> None:
    """Set up INFO-level timestamped logging for CLI runs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def first_line(doc: str | None) -> str | None:
    """Return the first line of a module docstring, for use as a CLI description."""
    return doc.strip().splitlines()[0] if doc and doc.strip() else None


def run_cli(spec: DatasetSpec, doc: str | None, argv: list[str] | None = None) -> None:
    """CLI entry point shared by the per-season loaders; ``doc`` is the module docstring."""
    configure_logging()
    args = parse_season_args(build_parser(first_line(doc), spec.default_out_dir), argv)
    seasons = list(range(args.start_season, args.end_season + 1))
    ingest_seasons(spec, seasons, args.out_dir, force=args.force)
