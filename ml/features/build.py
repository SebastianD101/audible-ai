"""Build the feature table: one row per (player, target week) with features and labels.

Usage:
    uv run python -m ml.features.build --start-season 2018 --end-season 2025

Output (gitignored, rebuilt deterministically from the raw layer)::

    ml/data/processed/features/season=YYYY.parquet
    ml/data/processed/features/manifest.json

The manifest lists feature and label columns, row counts and SHA-256 per file, the git commit,
and the SHA-256 of every raw manifest it was built from, so a model run can be traced back to
exact inputs. Each target week is built independently through the as-of rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ml.features.asof import Week
from ml.features.availability import availability_features
from ml.features.context import context_features
from ml.features.defense import defense_features
from ml.features.kickoff import kickoff_features
from ml.features.load import DEFAULT_RAW_DIR, RawData, load_raw
from ml.features.spine import LABEL_COLUMNS, SPINE_COLUMNS, add_labels, fixtures, spine
from ml.features.usage import usage_features
from ml.ingest import common

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("ml/data/processed/features")
RAW_MANIFESTS = [
    "player_stats/manifest.json",
    "schedules/manifest.json",
    "rosters_weekly/manifest.json",
    "injuries/manifest.json",
    "snap_counts/manifest.json",
    "players/manifest.json",
]


def week_features(raw: RawData, as_of: Week) -> pl.DataFrame:
    """Return the spine joined with every feature for one target week (no labels).

    ``ko_*`` columns follow the kickoff cutoff and exist only for the ablation (see
    ``ml.features.kickoff``); everything else follows the waiver-time cutoff.
    """
    fx = fixtures(raw.games)
    rows = spine(raw.stats, raw.rosters, fx, as_of)
    if rows.is_empty():
        return rows
    return (
        rows.join(usage_features(raw.stats, raw.snaps, as_of), on="player_id", how="left")
        .join(defense_features(raw.stats, as_of), on=["opponent", "position"], how="left")
        .join(context_features(fx, raw.games, as_of), on="team", how="left")
        .join(
            availability_features(
                rows, raw.stats, raw.snaps, raw.rosters, raw.injuries, raw.games, as_of
            ),
            on="player_id",
            how="left",
        )
        .join(kickoff_features(rows, raw.lines, raw.injuries, as_of), on="player_id", how="left")
    )


def build_week(raw: RawData, as_of: Week) -> pl.DataFrame:
    """Return spine, features, and labels for one target week (empty if no spine)."""
    features = week_features(raw, as_of)
    return add_labels(features, raw.stats) if features.height else features


def target_weeks(raw: RawData, season: int) -> list[Week]:
    """Every regular-season week of ``season`` that has a spine (week 2 onward)."""
    weeks = raw.games.filter(pl.col("season") == season)["week"].unique().sort().to_list()
    return [Week(season, w) for w in weeks if w >= 2]


def build_season(raw: RawData, season: int) -> pl.DataFrame:
    """Build every target week of a season into one frame."""
    frames = [build_week(raw, w) for w in target_weeks(raw, season)]
    frames = [f for f in frames if f.height]
    if not frames:
        raise ValueError(f"season {season}: no target weeks with data")
    return pl.concat(frames, how="diagonal_relaxed").sort("week", "player_id")


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Model inputs: every column that is neither an identifier nor a label."""
    return [c for c in df.columns if c not in SPINE_COLUMNS and c not in LABEL_COLUMNS]


def git_commit() -> str:
    """Return the current commit SHA (suffixed ``-dirty`` with uncommitted changes)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def raw_manifest_hashes(raw_dir: Path) -> dict[str, str | None]:
    """SHA-256 of each raw manifest (None if missing): the input data version."""
    return {
        name: hashlib.sha256((raw_dir / name).read_bytes()).hexdigest()
        if (raw_dir / name).exists()
        else None
        for name in RAW_MANIFESTS
    }


def write_features(
    raw: RawData, seasons: list[int], out_dir: Path, raw_dir: Path
) -> dict[str, object]:
    """Build and write each season, then the manifest; return the manifest."""
    files = []
    columns: list[str] = []
    for season in seasons:
        df = build_season(raw, season)
        path = common.write_parquet_atomic(df, out_dir / f"season={season}.parquet")
        columns = columns or feature_columns(df)
        files.append(
            {
                "season": season,
                "file": path.name,
                "rows": df.height,
                "sha256": common.sha256_file(path),
            }
        )
        logger.info(
            "features %d: %d rows, %d features", season, df.height, len(feature_columns(df))
        )
    manifest: dict[str, object] = {
        "dataset": "features",
        "information_cutoff": "waiver time: rows from weeks before the target week only",
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "raw_manifest_sha256": raw_manifest_hashes(raw_dir),
        "id_columns": SPINE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "feature_columns": columns,
        "seasons": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=common.first_line(__doc__))
    parser.add_argument("--start-season", type=int, default=common.DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=common.DEFAULT_END_SEASON)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return common.parse_season_args(parser, argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.configure_logging()
    args = parse_args(argv)
    seasons = list(range(args.start_season, args.end_season + 1))
    write_features(load_raw(args.raw_dir), seasons, args.out_dir, args.raw_dir)


if __name__ == "__main__":
    main()
