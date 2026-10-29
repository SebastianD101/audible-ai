"""Project next week's PPR points for every spine player of a live season.

Usage:
    uv run python -m ml.serve.project --season 2026 --week 8
    uv run python -m ml.serve.project --season 2026 --week auto   # the scheduled refresh

Run on Tuesday night after week W-1's games, the same waiver-time cutoff the features use.
The model is refit from scratch, as the backtest does each week:

- training rows: the processed feature table for past seasons, plus this season's weeks
  2..W-1 built now from the raw layer, with labels from games already played
- target rows: week W features (no labels; the games haven't happened)
- params: the committed serving params (``ml.serve.params``), tuned without this season

So the live season trains refits but never tunes anything, and each week's file is written
before its games, which keeps a later live evaluation honest. Writes::

    ml/data/projections/season=YYYY/week=WW.parquet
    ml/data/projections/season=YYYY/week=WW.json     manifest: data versions, params, git SHA

``hold_value`` is the season-to-date average, the longer-horizon value the optimizer's drop
guard uses so a one-week projection never recommends dropping an injured starter.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import mlflow
import polars as pl

from ml.backtest.predictors import SeasonAvgBaseline
from ml.backtest.run import DEFAULT_ARTIFACT_DIR, DEFAULT_TRACKING_URI, mlflow_experiment
from ml.features.asof import Week, before_expr
from ml.features.build import DEFAULT_OUT_DIR as FEATURES_DIR
from ml.features.build import build_week, git_commit, raw_manifest_hashes, week_features
from ml.features.load import DEFAULT_RAW_DIR, RawData, load_raw
from ml.ingest import common
from ml.models.lgbm import LgbmParams, LgbmPredictor
from ml.serve.params import DEFAULT_PARAMS_DIR, params_path, read_params

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("ml/data/projections")
OUTPUT_COLUMNS = ["player_id", "name", "position", "season", "week", "team", "opponent"]


class StaleDataError(ValueError):
    """The raw layer doesn't yet have every result the target week's features need."""


def check_ready(games: pl.DataFrame, target: Week) -> None:
    """Require week ``target`` on the schedule and every week W-1 game to have a final score.

    Raises:
        StaleDataError: with what is missing and how to refresh it.
    """
    season = games.filter(pl.col("season") == target.season)
    if season.filter(pl.col("week") == target.week).is_empty():
        raise StaleDataError(f"{target}: not on the schedule")
    unfinished = season.filter((pl.col("week") == target.week - 1) & pl.col("points_for").is_null())
    if unfinished.height:
        raise StaleDataError(
            f"{target}: {unfinished['game_id'].n_unique()} week {target.week - 1} game(s) have "
            "no final score; re-ingest schedules and player_stats after the last game"
        )


def next_target_week(games: pl.DataFrame, season: int) -> Week | None:
    """The first week of ``season`` with a game still unplayed, or None once the season is over.

    The scheduled refresh projects this week. A week whose games have started is never
    returned as projectable (see ``auto_target``), so a late run can't see its results.
    """
    open_weeks = games.filter((pl.col("season") == season) & pl.col("points_for").is_null())
    if open_weeks.is_empty():
        return None
    return Week(season, int(open_weeks["week"].min()))  # type: ignore[arg-type]


def auto_target(games: pl.DataFrame, season: int, out_dir: Path) -> tuple[Week | None, str]:
    """Return (the week to project now, or None with the reason there's nothing to do).

    Nothing is projected after the season, in week 1, once any game of the week has a score
    (projections must predate kickoff), or when the week's file already exists (a published
    projection is never overwritten).
    """
    target = next_target_week(games, season)
    if target is None:
        return None, f"{season}: the season is over"
    if target.week < 2:
        return None, f"{target}: week 1 is never projected"
    started = games.filter(
        (pl.col("season") == season)
        & (pl.col("week") == target.week)
        & pl.col("points_for").is_not_null()
    )
    if started.height:
        return None, f"{target}: games have already kicked off; too late to project"
    if output_base(out_dir, target).with_suffix(".parquet").exists():
        return None, f"{target}: already projected"
    return target, ""


def training_rows(past: pl.DataFrame, raw: RawData, target: Week) -> pl.DataFrame:
    """Past-season feature rows plus this season's labeled weeks before ``target``.

    Raises:
        ValueError: if there are no training rows.
    """
    current = [build_week(raw, Week(target.season, w)) for w in range(2, target.week)]
    frames = [past.filter(pl.col("season") < target.season), *(f for f in current if f.height)]
    train = pl.concat(frames, how="diagonal_relaxed")
    if train.is_empty():
        raise ValueError(f"{target}: no training rows")
    assert train.filter(~before_expr(target)).is_empty(), "training rows must precede target"
    return train


def project_week(
    past: pl.DataFrame, raw: RawData, target: Week, params: LgbmParams
) -> tuple[pl.DataFrame, int]:
    """Return (one projection row per spine player for ``target``, training row count).

    Raises:
        StaleDataError: if week W-1 isn't complete in ``raw``.
        ValueError: if ``target`` has no spine (week 1) or no training rows.
    """
    check_ready(raw.games, target)
    target_rows = week_features(raw, target)
    if target_rows.is_empty():
        raise ValueError(f"{target}: no spine rows (week 1 is never projected)")
    train = training_rows(past, raw, target)
    model = LgbmPredictor(params)
    model.fit(train)
    out = target_rows.select(OUTPUT_COLUMNS).with_columns(
        model.predict(target_rows).alias("projection"),
        SeasonAvgBaseline().predict(target_rows).clip(0.0).alias("hold_value"),
    )
    return out.sort("projection", descending=True), train.height


def output_base(out_dir: Path, target: Week) -> Path:
    """The week's output path without a suffix (``.parquet`` data, ``.json`` manifest)."""
    return out_dir / f"season={target.season}" / f"week={target.week:02d}"


def write_projections(
    projections: pl.DataFrame, out_dir: Path, target: Week, info: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Write the week's Parquet and manifest; return (parquet path, manifest)."""
    base = output_base(out_dir, target)
    path = common.write_parquet_atomic(projections, base.with_suffix(".parquet"))
    digest = common.sha256_file(path)
    manifest = {
        "season": target.season,
        "week": target.week,
        "version": digest[:12],
        "sha256": digest,
        "rows": projections.height,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **info,
    }
    base.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path, manifest


def week_arg(value: str) -> int | Literal["auto"]:
    """Parse ``--week``: a number or ``auto``."""
    if value == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a week number or 'auto', got {value!r}"
        ) from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments; only live seasons are projected here."""
    parser = argparse.ArgumentParser(description=common.first_line(__doc__))
    parser.add_argument("--season", type=int, default=common.HOLDOUT_SEASON)
    parser.add_argument(
        "--week", type=week_arg, required=True, help="a week number, or 'auto' (next unplayed)"
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--params-dir", type=Path, default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args(argv)
    if args.season < common.HOLDOUT_SEASON:
        parser.error("past seasons are evaluated by the backtest, not projected live")
    if args.week != "auto" and args.week < 2:
        parser.error("week 1 has no in-season data and is never projected")
    return args


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.configure_logging()
    args = parse_args(argv)
    raw = load_raw(args.raw_dir)
    if args.week == "auto":
        auto, reason = auto_target(raw.games, args.season, args.out_dir)
        if auto is None:
            logger.info("nothing to project: %s", reason)
            return
        target = auto
    else:
        target = Week(args.season, args.week)
    ppath = params_path(args.season, args.params_dir)
    params = read_params(ppath)
    past = pl.scan_parquet(args.features_dir / "season=*.parquet").filter(
        pl.col("season") < args.season
    )
    projections, n_train = project_week(past.collect(), raw, target, params)

    commit = git_commit()
    features_manifest = args.features_dir / "manifest.json"
    info = {
        "model": "lgbm",
        "params": asdict(params),
        "params_sha256": common.sha256_file(ppath),
        "train_rows": n_train,
        "git_commit": commit,
        "features_manifest_sha256": common.sha256_file(features_manifest)
        if features_manifest.exists()
        else "missing",
        "raw_manifest_sha256": raw_manifest_hashes(args.raw_dir),
    }
    path, manifest = write_projections(projections, args.out_dir, target, info)

    experiment_id = mlflow_experiment(args.tracking_uri, args.artifact_dir)
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"serve-{target}-{commit[:7]}"):
        mlflow.set_tags({"git_commit": commit, "kind": "serve-projections"})
        mlflow.log_params(
            {"season": target.season, "week": target.week, "train_rows": n_train}
            | {f"lgbm.{k}": v for k, v in asdict(params).items()}
        )
        mlflow.set_tag("projections_sha256", manifest["sha256"])
        mlflow.log_artifact(str(path.with_suffix(".json")))
    logger.info("%s: %d projections written to %s", target, projections.height, path)


if __name__ == "__main__":
    main()
