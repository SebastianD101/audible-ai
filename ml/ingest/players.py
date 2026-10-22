"""Download the nflverse player ID crosswalk and store it as a single Parquet snapshot.

Usage:
    uv run python -m ml.ingest.players [--force]

Output: ``ml/data/raw/players/players.parquet`` plus ``manifest.json``.

``nflreadpy.load_players()`` is a *current* snapshot of every player, not a per-season table.
Its team, status, and position fields describe today, so using them for a past week would leak
future information. Only ID columns and the display name are stored, which makes that leak
impossible downstream. The main use is mapping snap counts (``pfr_player_id``) to gsis IDs;
weekly rosters alone cover only ~56-66% of offensive snap rows, this table ~99.7-100%.

``gsis_id`` and ``pfr_id`` must be unique (among non-null), so joins on either cannot fan out.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import nflreadpy
import polars as pl

from ml.ingest import common
from ml.ingest import snap_counts as sc

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("ml/data/raw/players")
FILE_NAME = "players.parquet"
ID_COLUMNS = ["gsis_id", "pfr_id", "espn_id", "nfl_id", "pff_id", "esb_id", "smart_id", "otc_id"]
KEPT_COLUMNS = ["gsis_id", "display_name", *ID_COLUMNS[1:]]
UNIQUE_COLUMNS = ["gsis_id", "pfr_id"]
MIN_SNAP_COVERAGE = 0.99


def fetch_players() -> pl.DataFrame:
    """Download the current nflverse players table."""
    df: pl.DataFrame = nflreadpy.load_players()
    return df


def validate_players(df: pl.DataFrame) -> pl.DataFrame:
    """Check columns, types, non-null and unique IDs; return only the kept columns.

    Raises:
        common.DataValidationError: If any check fails.
    """
    missing = sorted(set(KEPT_COLUMNS) - set(df.columns))
    if missing:
        raise common.DataValidationError(f"players: missing columns {missing}")
    wrong_types = {c: str(df.schema[c]) for c in KEPT_COLUMNS if df.schema[c] != pl.String}
    if wrong_types:
        raise common.DataValidationError(f"players: unexpected column types {wrong_types}")
    if df.is_empty():
        raise common.DataValidationError("players: no rows returned")
    if df["gsis_id"].null_count():
        raise common.DataValidationError("players: null gsis_id")
    for col in UNIQUE_COLUMNS:
        ids = df[col].drop_nulls()
        if ids.len() != ids.n_unique():
            raise common.DataValidationError(
                f"players: {ids.len() - ids.n_unique()} duplicate {col}"
            )
    return df.select(KEPT_COLUMNS)


def snap_coverage(players: pl.DataFrame, snaps: pl.DataFrame) -> float:
    """Share of snap-count rows with offensive snaps whose PFR ID maps to a gsis ID."""
    offense = snaps.filter(pl.col("offense_snaps") > 0)
    if offense.is_empty():
        return 1.0
    known = players["pfr_id"].drop_nulls().implode()
    share: float = offense.select(pl.col("pfr_player_id").is_in(known).mean()).item()
    return share


def write_manifest(out_dir: Path, path: Path, df: pl.DataFrame) -> Path:
    """Write ``manifest.json`` for the snapshot (fetch time and hash are the data version)."""
    manifest = {
        "dataset": "players",
        "source": "nflverse players via nflreadpy.load_players (current snapshot, IDs only)",
        "nflreadpy_version": version("nflreadpy"),
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "file": path.name,
        "rows": df.height,
        "columns": df.columns,
        "sha256": common.sha256_file(path),
    }
    out = out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def ingest_players(
    out_dir: Path = DEFAULT_OUT_DIR,
    snap_dir: Path = sc.DEFAULT_OUT_DIR,
    *,
    force: bool = False,
) -> Path:
    """Fetch, validate, and store the crosswalk; warn if stored snap counts map poorly."""
    path = out_dir / FILE_NAME
    if path.exists() and not force:
        logger.info("players: %s exists, skipping (use --force to refresh)", path)
    else:
        logger.info("players: downloading")
        df = validate_players(fetch_players())
        common.write_parquet_atomic(df, path)
        write_manifest(out_dir, path, df)
        logger.info("players: %d rows in %s", df.height, path)

    snap_files = sorted(snap_dir.glob("season=*.parquet"))
    if snap_files:
        players = pl.read_parquet(path)
        for snap_file in snap_files:
            snaps = pl.read_parquet(snap_file, columns=["pfr_player_id", "offense_snaps"])
            coverage = snap_coverage(players, snaps)
            log = logger.warning if coverage < MIN_SNAP_COVERAGE else logger.info
            log(
                "players: %s offensive snap rows mapped to gsis: %.2f%%",
                snap_file.stem,
                100 * coverage,
            )
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=common.first_line(__doc__))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--snap-dir", type=Path, default=sc.DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="re-download the snapshot")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.configure_logging()
    args = parse_args(argv)
    ingest_players(args.out_dir, args.snap_dir, force=args.force)


if __name__ == "__main__":
    main()
