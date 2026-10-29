"""Score the committed live projections against what happened (the 2026 holdout).

Usage:
    uv run python -m ml.serve.evaluate --season 2026

Each ``week=WW.parquet`` written by ``ml.serve.project`` is joined to that week's actual PPR
points once every game of the week has a final score. LightGBM's ``projection`` is compared
with ``hold_value``, which is the season-average baseline computed at the same waiver-time
cutoff, using the backtest's metrics and week-block bootstrap (``ml.backtest.metrics``).

This only reads: nothing here feeds back into params or features, so 2026 stays an untuned
test set. A week is excluded if its manifest says it was generated after the week's first
kickoff, because such a file could have seen results. Writes ``docs/live-<season>.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import polars as pl

from ml.backtest.metrics import METRICS, Estimate, bootstrap, paired_bootstrap
from ml.backtest.run import DEFAULT_ARTIFACT_DIR, DEFAULT_TRACKING_URI, mlflow_experiment
from ml.backtest.scopes import RELEVANT_PER_POSITION
from ml.features.build import git_commit
from ml.features.load import DEFAULT_RAW_DIR, load_raw
from ml.features.spine import add_labels
from ml.ingest import common
from ml.serve.project import DEFAULT_OUT_DIR as PROJECTIONS_DIR

logger = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE")
MODELS = {"lgbm": "projection", "season_avg": "hold_value"}
NFL_TIMEZONE = "America/New_York"  # nflverse gametime is US Eastern


@dataclass(frozen=True)
class Excluded:
    """A projected week left out of the evaluation, and why."""

    week: int
    reason: str


def first_kickoffs(schedules: pl.DataFrame) -> dict[int, datetime]:
    """Each week's earliest kickoff in UTC, from nflverse ``gameday`` and ``gametime``."""
    kickoffs = (
        schedules.filter(pl.col("game_type") == "REG")
        .with_columns(
            (pl.col("gameday") + " " + pl.col("gametime"))
            .str.to_datetime("%Y-%m-%d %H:%M", time_zone=NFL_TIMEZONE)
            .dt.convert_time_zone("UTC")
            .alias("kickoff")
        )
        .group_by("week")
        .agg(pl.col("kickoff").min())
    )
    return dict(zip(kickoffs["week"].to_list(), kickoffs["kickoff"].to_list(), strict=True))


def final_weeks(games: pl.DataFrame, season: int) -> set[int]:
    """Weeks of ``season`` in which every game has a final score."""
    by_week = (
        games.filter(pl.col("season") == season)
        .group_by("week")
        .agg(pl.col("points_for").is_not_null().all().alias("final"))
    )
    return set(by_week.filter(pl.col("final"))["week"].to_list())


def scorable_weeks(
    manifests: Mapping[int, dict[str, object]],
    final: set[int],
    kickoffs: Mapping[int, datetime],
) -> tuple[list[int], list[Excluded]]:
    """Split projected weeks into those to score and those excluded (with reasons).

    Weeks not yet final are skipped silently: they'll be scored on a later run.
    """
    keep, excluded = [], []
    for week, manifest in sorted(manifests.items()):
        if week not in final:
            continue
        generated = datetime.fromisoformat(str(manifest["generated_at_utc"]))
        kickoff = kickoffs.get(week)
        if kickoff is None or generated >= kickoff:
            excluded.append(Excluded(week, "generated after the week's first kickoff"))
            continue
        keep.append(week)
    return keep, excluded


def relevant(projections: pl.DataFrame) -> pl.Series:
    """The backtest's ``relevant`` scope: top N per (week, position) by pre-week season PPG.

    ``hold_value`` is that same pre-week score (season PPG with the prior-season fallback),
    so this matches ``ml.backtest.scopes.relevance`` without needing the feature columns.
    """
    limit = pl.col("position").replace_strict(
        dict(RELEVANT_PER_POSITION), default=0, return_dtype=pl.Int64
    )
    ranked = projections.with_row_index("_row").sort(
        "hold_value", "player_id", descending=[True, False]
    )
    rank = pl.int_range(1, pl.len() + 1).over("season", "week", "position")
    flags = ranked.select("_row", (rank <= limit).alias("relevant")).sort("_row")
    return flags["relevant"]


def as_predictions(labeled: pl.DataFrame, column: str) -> pl.DataFrame:
    """The backtest metrics' input shape for one model."""
    return labeled.select(
        "player_id", "season", "week", "position", "label_ppr", pl.col(column).alias("prediction")
    )


def evaluate(labeled: pl.DataFrame) -> dict[str, dict[str, Estimate]]:
    """Estimates keyed like ``relevant.ALL.lgbm`` and ``relevant.ALL.lgbm-season_avg``."""
    out: dict[str, dict[str, Estimate]] = {}
    for scope in ("relevant", "all"):
        in_scope = labeled.filter(pl.col("relevant")) if scope == "relevant" else labeled
        for position in ("ALL", *POSITIONS):
            rows = (
                in_scope if position == "ALL" else in_scope.filter(pl.col("position") == position)
            )
            if rows.is_empty():
                continue
            preds = {m: as_predictions(rows, c) for m, c in MODELS.items()}
            for model, p in preds.items():
                out[f"{scope}.{position}.{model}"] = bootstrap(p)
            out[f"{scope}.{position}.lgbm-season_avg"] = paired_bootstrap(
                preds["lgbm"], preds["season_avg"]
            )
    return out


def fmt(e: Estimate, digits: int = 2) -> str:
    """``1.23 [1.01, 1.45]``."""
    return f"{e.point:.{digits}f} [{e.lo:.{digits}f}, {e.hi:.{digits}f}]"


def report(
    season: int,
    weeks: list[int],
    excluded: list[Excluded],
    results: dict[str, dict[str, Estimate]],
) -> str:
    """The Markdown report for ``docs/live-<season>.md``."""
    lines = [
        f"# Live {season} results",
        "",
        "<!-- Generated by `uv run python -m ml.serve.evaluate`; don't edit by hand. -->",
        "",
        f"{season} is the held-out live test set: projections are committed to git before each",
        "week's games (`ml/data/projections/`), and nothing is tuned on this season. LightGBM is",
        "compared with the season-average baseline at the same waiver-time cutoff. Intervals are",
        "95% week-block bootstrap intervals, as in the backtest.",
        "",
    ]
    if excluded:
        lines += [f"- Week {e.week} excluded: {e.reason}." for e in excluded] + [""]
    if not weeks:
        lines += ["No projected week has finished yet, so there is nothing to score.", ""]
        return "\n".join(lines)
    n_rows = results["relevant.ALL.lgbm"]["rmse"].n_rows
    lines += [
        f"**Sample: {len(weeks)} week(s) ({', '.join(map(str, weeks))}), {n_rows} relevant "
        "player-weeks.**",
    ]
    if len(weeks) < 4:
        lines += [
            "",
            "This is far too few weeks to conclude anything: weekly fantasy scoring is noisy, and",
            "with this few weeks the bootstrap intervals are themselves unreliable. Read these as",
            "a tracker, not a result.",
        ]
    lines += [
        "",
        "## Relevant players (the backtest's headline scope)",
        "",
        "| Position | Model | RMSE | MAE | Spearman |",
        "| --- | --- | --- | --- | --- |",
    ]
    for position in ("ALL", *POSITIONS):
        for model in (*MODELS, "lgbm-season_avg"):
            est = results.get(f"relevant.{position}.{model}")
            if est is None:
                continue
            label = "LightGBM minus season avg" if model == "lgbm-season_avg" else model
            lines.append(
                f"| {position} | {label} | " + " | ".join(fmt(est[m]) for m in METRICS) + " |"
            )
    lines += [
        "",
        "Lower RMSE and MAE are better and higher Spearman is better, so a negative RMSE",
        "difference favors LightGBM.",
        "",
    ]
    return "\n".join(lines)


def read_manifests(projections_dir: Path, season: int) -> dict[int, dict[str, object]]:
    """Every week's manifest for ``season``, keyed by week."""
    folder = projections_dir / f"season={season}"
    manifests = {}
    for path in sorted(folder.glob("week=*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifests[int(manifest["week"])] = manifest
    return manifests


def labeled_projections(
    projections_dir: Path, season: int, weeks: list[int], stats: pl.DataFrame
) -> pl.DataFrame:
    """The scored weeks' projections with actual PPR points and the ``relevant`` flag."""
    folder = projections_dir / f"season={season}"
    frames = [pl.read_parquet(folder / f"week={w:02d}.parquet") for w in weeks]
    projections = pl.concat(frames, how="diagonal_relaxed")
    labeled = add_labels(projections, stats)
    return labeled.with_columns(relevant(labeled))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=common.first_line(__doc__))
    parser.add_argument("--season", type=int, default=common.HOLDOUT_SEASON)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--projections-dir", type=Path, default=PROJECTIONS_DIR)
    parser.add_argument("--out", type=Path, default=None, help="default: docs/live-<season>.md")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.configure_logging()
    args = parse_args(argv)
    raw = load_raw(args.raw_dir)
    schedules = pl.read_parquet(args.raw_dir / "schedules" / f"season={args.season}.parquet")
    manifests = read_manifests(args.projections_dir, args.season)
    weeks, excluded = scorable_weeks(
        manifests, final_weeks(raw.games, args.season), first_kickoffs(schedules)
    )
    for skipped in excluded:
        logger.warning("week %d excluded: %s", skipped.week, skipped.reason)
    results: dict[str, dict[str, Estimate]] = {}
    if weeks:
        labeled = labeled_projections(args.projections_dir, args.season, weeks, raw.stats)
        results = evaluate(labeled)
    out = args.out or Path("docs") / f"live-{args.season}.md"
    out.write_text(report(args.season, weeks, excluded, results), encoding="utf-8")

    commit = git_commit()
    experiment_id = mlflow_experiment(args.tracking_uri, args.artifact_dir)
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"live-{args.season}-{commit[:7]}"):
        mlflow.set_tags({"git_commit": commit, "kind": "live-evaluation"})
        mlflow.log_params({"season": args.season, "weeks": ",".join(map(str, weeks)) or "none"})
        for key, estimates in results.items():
            for metric, e in estimates.items():
                mlflow.log_metrics(
                    {f"{key}.{metric}": e.point, f"{key}.{metric}_lo": e.lo}
                    | {f"{key}.{metric}_hi": e.hi}
                )
        mlflow.log_artifact(str(out))
    logger.info("%d week(s) scored, %d excluded; wrote %s", len(weeks), len(excluded), out)


if __name__ == "__main__":
    main()
