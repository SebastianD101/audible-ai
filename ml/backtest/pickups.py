"""Historical pickup backtest: points actually gained by acting on each model's swaps.

For every evaluation week, a 12-team league is simulated from information available before
kickoff, and the optimizer (``ml.optimize``) recommends add/drop swaps for sampled rosters
using each projection source. Realized value is the actual PPR points of the lineup the
manager would start.

- **Ownership proxy:** the ``n_teams * roster_size`` QB/RB/WR/TE with the highest
  ``season_avg`` prediction are rostered; everyone else in the week's spine is a free agent.
  ``season_avg`` uses only games before W and belongs to neither model being compared.
- **Rosters:** sampled (seeded) from the rostered pool with a fixed position composition.
- **Sources:** ``lgbm`` and ``last3`` projections from a walk-forward backtest run (already
  out-of-sample), plus ``oracle`` (actual points) as an upper bound for scale. Every source
  uses ``season_avg`` as the hold value, so no source can drop a season-long starter.
- **Outcomes per (week, roster, source, max_moves):** realized points of the projected-best
  lineup before and after the recommended moves, and their difference (pickup gain).

The 2026 holdout season is rejected; this module never reads it.

Usage:
    uv run python -m ml.backtest.pickups --start-season 2020 --end-season 2025

Writes ``ml/data/backtest/pickups-<timestamp>-<commit>/`` (``results.parquet``,
``metrics.parquet``, ``metrics.md``) and one MLflow run.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from ml.backtest.metrics import DEFAULT_BOOTSTRAP, DEFAULT_SEED, Estimate, week_draws
from ml.backtest.run import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_TRACKING_URI,
    mlflow_experiment,
)
from ml.features.build import git_commit
from ml.ingest import common
from ml.optimize.league import LeagueSettings, Player
from ml.optimize.lineup import Lineup
from ml.optimize.swaps import recommend

logger = logging.getLogger(__name__)

RUN_PREFIX = "pickups-"
MODEL_SOURCES = ("lgbm", "last3")
ORACLE = "oracle"
OWNERSHIP_SOURCE = "season_avg"
DEFAULT_COMPOSITION: Mapping[str, int] = {"QB": 2, "RB": 4, "WR": 5, "TE": 2}
DEFAULT_TEAMS = 12
DEFAULT_ROSTERS_PER_WEEK = 50
DEFAULT_MAX_MOVES = (1, 2)


@dataclass(frozen=True)
class SimConfig:
    """Everything that defines one pickup backtest, for reproducibility and logging."""

    n_teams: int = DEFAULT_TEAMS
    rosters_per_week: int = DEFAULT_ROSTERS_PER_WEEK
    max_moves: tuple[int, ...] = DEFAULT_MAX_MOVES
    min_gain: float = 0.5
    seed: int = DEFAULT_SEED

    @property
    def roster_size(self) -> int:
        """Players per simulated roster."""
        return sum(DEFAULT_COMPOSITION.values())


def wide_predictions(preds: pl.DataFrame, start_season: int, end_season: int) -> pl.DataFrame:
    """One row per player-week with a ``pred_<model>`` column per needed model.

    Raises:
        ValueError: if a season is in the holdout or a needed model is missing.
    """
    if end_season >= common.HOLDOUT_SEASON:
        raise ValueError(f"{common.HOLDOUT_SEASON}+ is the live holdout; backtests stop at 2025")
    needed = [*MODEL_SOURCES, OWNERSHIP_SOURCE]
    missing = sorted(set(needed) - set(preds["model"].unique().to_list()))
    if missing:
        raise ValueError(f"predictions are missing models {missing}")
    keys = ["player_id", "position", "season", "week", "label_ppr"]
    return (
        preds.filter(
            pl.col("season").is_between(start_season, end_season), pl.col("model").is_in(needed)
        )
        .select(*keys, "model", "prediction")
        .pivot(on="model", index=keys, values="prediction")
        .rename({m: f"pred_{m}" for m in needed})
        .sort("season", "week", "player_id")
    )


def rostered_ids(week: pl.DataFrame, n_rostered: int) -> pl.Series:
    """IDs of the ``n_rostered`` players with the highest ownership-proxy prediction."""
    return week.sort([f"pred_{OWNERSHIP_SOURCE}", "player_id"], descending=[True, False]).head(
        n_rostered
    )["player_id"]


def sample_rosters(
    rostered: pl.DataFrame,
    n: int,
    composition: Mapping[str, int],
    rng: np.random.Generator,
) -> list[list[Player]]:
    """Draw ``n`` rosters from the rostered pool, each with ``composition`` per position.

    Rosters are independent draws (two may share players): each is one plausible team.

    Raises:
        ValueError: if the pool has too few players at some position.
    """
    by_position = {
        pos: sorted(rostered.filter(pl.col("position") == pos)["player_id"].to_list())
        for pos in composition
    }
    for pos, count in composition.items():
        if len(by_position[pos]) < count:
            raise ValueError(f"rostered pool has {len(by_position[pos])} {pos}, need {count}")
    return [
        [
            Player(str(pid), pos)
            for pos, count in composition.items()
            for pid in rng.choice(by_position[pos], size=count, replace=False)
        ]
        for _ in range(n)
    ]


def realized(lineup: Lineup, actual: Mapping[str, float]) -> float:
    """Actual points scored by a lineup's starters."""
    return sum(actual[pid] for pid in lineup.player_ids)


def simulate_week(
    week: pl.DataFrame,
    settings: LeagueSettings,
    config: SimConfig,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Outcomes for every sampled roster, source, and move limit in one week."""
    n_rostered = config.n_teams * config.roster_size
    owned = set(rostered_ids(week, n_rostered).to_list())
    rostered = week.filter(pl.col("player_id").is_in(owned))
    free_agents = [
        Player(row["player_id"], row["position"])
        for row in week.filter(~pl.col("player_id").is_in(owned))
        .select("player_id", "position")
        .iter_rows(named=True)
    ]
    actual = dict(zip(week["player_id"], week["label_ppr"], strict=True))
    hold = dict(zip(week["player_id"], week[f"pred_{OWNERSHIP_SOURCE}"], strict=True))
    projections = {
        **{s: dict(zip(week["player_id"], week[f"pred_{s}"], strict=True)) for s in MODEL_SOURCES},
        ORACLE: actual,
    }

    rows = []
    season, wk = int(week["season"][0]), int(week["week"][0])
    rosters = sample_rosters(rostered, config.rosters_per_week, DEFAULT_COMPOSITION, rng)
    for r, roster in enumerate(rosters):
        for source, proj in projections.items():
            for k in config.max_moves:
                rec = recommend(
                    roster,
                    free_agents,
                    proj,
                    settings,
                    max_moves=k,
                    hold_values=hold,
                    min_gain=config.min_gain,
                )
                before, after = realized(rec.before, actual), realized(rec.after, actual)
                rows.append(
                    {
                        "season": season,
                        "week": wk,
                        "roster": r,
                        "source": source,
                        "max_moves": k,
                        "moves": len(rec.adds),
                        "projected_gain": rec.projected_gain,
                        "realized_before": before,
                        "realized_after": after,
                        "realized_gain": after - before,
                    }
                )
    return pl.DataFrame(rows)


def simulate(wide: pl.DataFrame, settings: LeagueSettings, config: SimConfig) -> pl.DataFrame:
    """Run ``simulate_week`` over every (season, week) in ``wide``, with one seeded RNG."""
    rng = np.random.default_rng(config.seed)
    frames = []
    for (season, week_no), week in wide.group_by(["season", "week"], maintain_order=True):
        frames.append(simulate_week(week, settings, config, rng))
        logger.info("pickups: simulated %s week %s", season, week_no)
    return pl.concat(frames)


def weekly_sums(results: pl.DataFrame, value: pl.Expr) -> pl.DataFrame:
    """Per-week sum of ``value`` and row count, in a fixed week order."""
    return (
        results.group_by("season", "week")
        .agg(value.sum().alias("total"), pl.len().alias("n"))
        .sort("season", "week")
    )


def week_mean_estimate(
    metric: str,
    weekly: pl.DataFrame,
    n_rows: int,
    draws: np.ndarray,
) -> Estimate:
    """Mean of a per-row quantity with a week-block bootstrap interval.

    Rows are pooled within each draw (sum of totals over sum of counts), so weeks with more
    rows weigh more, exactly as in the point estimate. Draws with no rows are skipped.
    """
    total, n = weekly["total"].to_numpy(), weekly["n"].to_numpy().astype(np.float64)
    if n.sum() == 0:
        return Estimate(metric, float("nan"), float("nan"), float("nan"), weekly.height, 0)
    point = float(total.sum() / n.sum())
    counts = draws @ n
    boot = (draws @ total)[counts > 0] / counts[counts > 0]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return Estimate(metric, point, float(lo), float(hi), weekly.height, n_rows)


def summarize(
    results: pl.DataFrame,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> pl.DataFrame:
    """Tidy table of estimates per (max_moves, source), plus paired lgbm - last3 differences.

    Metrics:

    - ``realized_gain``: actual points added by the recommended moves (0 when none).
    - ``realized_after``: actual points of the lineup started after acting.
    - ``projected_gain``: what the source expected the moves to add.
    - ``move_rate``: share of rosters given at least one move.
    - ``hit_rate``: among rosters given a move, share where it actually gained points.

    Week-block bootstrap: rosters in the same week share games, so whole weeks are resampled.
    The same draws are used for every row of the table, which makes differences paired.
    """
    n_weeks = results.select("season", "week").unique().height
    draws = week_draws(n_weeks, n_boot, seed)
    metrics = {
        "realized_gain": pl.col("realized_gain"),
        "realized_after": pl.col("realized_after"),
        "projected_gain": pl.col("projected_gain"),
        "move_rate": (pl.col("moves") > 0).cast(pl.Float64),
    }
    out = []
    for k in sorted(results["max_moves"].unique().to_list()):
        at_k = results.filter(pl.col("max_moves") == k)
        for source in [*MODEL_SOURCES, ORACLE]:
            rows = at_k.filter(pl.col("source") == source)
            for name, expr in metrics.items():
                est = week_mean_estimate(name, weekly_sums(rows, expr), rows.height, draws)
                out.append({"max_moves": k, "source": source, **est.__dict__})
            moved = rows.filter(pl.col("moves") > 0)
            hits = weekly_sums_aligned(moved, rows, (pl.col("realized_gain") > 0).cast(pl.Float64))
            est = week_mean_estimate("hit_rate", hits, moved.height, draws)
            out.append({"max_moves": k, "source": source, **est.__dict__})
        paired = at_k.filter(pl.col("source").is_in(MODEL_SOURCES)).pivot(
            on="source",
            index=["season", "week", "roster"],
            values=["realized_gain", "realized_after"],
        )
        for name in ("realized_gain", "realized_after"):
            diff = pl.col(f"{name}_lgbm") - pl.col(f"{name}_last3")
            est = week_mean_estimate(name, weekly_sums(paired, diff), paired.height, draws)
            out.append({"max_moves": k, "source": "lgbm - last3", **est.__dict__})
    return pl.DataFrame(out)


def weekly_sums_aligned(subset: pl.DataFrame, full: pl.DataFrame, value: pl.Expr) -> pl.DataFrame:
    """``weekly_sums`` of ``subset``, with zero rows for weeks it lacks, aligned to ``full``."""
    weeks = full.select("season", "week").unique()
    sums = weekly_sums(subset, value)
    return (
        weeks.join(sums, on=["season", "week"], how="left")
        .with_columns(pl.col("total").fill_null(0.0), pl.col("n").fill_null(0))
        .sort("season", "week")
    )


def fmt_table(table: pl.DataFrame, order: Sequence[str]) -> str:
    """Markdown table of point [lo, hi] per source and metric for one ``max_moves``."""
    lines = []
    for k in sorted(table["max_moves"].unique().to_list()):
        at_k = table.filter(pl.col("max_moves") == k)
        sources = at_k["source"].unique(maintain_order=True).to_list()
        lines += [f"### max_moves = {k}", "", "| metric | " + " | ".join(sources) + " |"]
        lines.append("| --- |" + " --- |" * len(sources))
        for metric in order:
            cells = []
            for s in sources:
                row = at_k.filter((pl.col("source") == s) & (pl.col("metric") == metric))
                if row.is_empty():
                    cells.append("")
                    continue
                r = row.row(0, named=True)
                cells.append(f"{r['point']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]")
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        n = at_k.filter(pl.col("metric") == "realized_gain").row(0, named=True)
        lines += ["", f"n = {n['n_rows']} rosters over {n['n_weeks']} weeks.", ""]
    return "\n".join(lines)


METRIC_ORDER = (
    "realized_gain",
    "realized_after",
    "projected_gain",
    "move_rate",
    "hit_rate",
)


def latest_predictions(out_dir: Path) -> Path:
    """The newest walk-forward run's ``predictions.parquet`` under ``out_dir``.

    Raises:
        FileNotFoundError: if no backtest run has written predictions yet.
    """
    runs = sorted(
        p for p in out_dir.glob("*/predictions.parquet") if not p.parent.name.startswith(RUN_PREFIX)
    )
    if not runs:
        raise FileNotFoundError(
            f"no predictions under {out_dir}; run `uv run python -m ml.backtest.run` first"
        )
    return runs[-1]


def flat_metrics(table: pl.DataFrame) -> dict[str, float]:
    """MLflow metric dict, e.g. ``k1.lgbm.realized_gain`` / ``_lo`` / ``_hi``."""
    out: dict[str, float] = {}
    for r in table.iter_rows(named=True):
        source = str(r["source"]).replace(" - ", "_minus_")
        key = f"k{r['max_moves']}.{source}.{r['metric']}"
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = r["point"], r["lo"], r["hi"]
    return {k: v for k, v in out.items() if v == v}  # drop NaN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments; rejects the holdout season."""
    parser = argparse.ArgumentParser(description=common.first_line(__doc__))
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=common.DEFAULT_END_SEASON)
    parser.add_argument("--predictions", type=Path, help="defaults to the newest backtest run")
    parser.add_argument("--rosters", type=int, default=DEFAULT_ROSTERS_PER_WEEK)
    parser.add_argument("--teams", type=int, default=DEFAULT_TEAMS)
    parser.add_argument("--max-moves", default="1,2", help="comma-separated move limits")
    parser.add_argument("--min-gain", type=float, default=SimConfig.min_gain)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = common.parse_season_args(parser, argv)
    if args.end_season >= common.HOLDOUT_SEASON:
        parser.error(f"{common.HOLDOUT_SEASON}+ is the live holdout; backtests stop at 2025")
    return args


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.configure_logging()
    args = parse_args(argv)
    config = SimConfig(
        n_teams=args.teams,
        rosters_per_week=args.rosters,
        max_moves=tuple(int(k) for k in args.max_moves.split(",")),
        min_gain=args.min_gain,
        seed=args.seed,
    )
    settings = LeagueSettings()
    source = args.predictions or latest_predictions(args.out_dir)
    commit = git_commit()
    run_id = f"{RUN_PREFIX}{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{commit[:7]}"
    logger.info("pickups %s: reading %s", run_id, source)

    wide = wide_predictions(pl.read_parquet(source), args.start_season, args.end_season)
    results = simulate(wide, settings, config)
    table = summarize(results, n_boot=args.bootstrap, seed=args.seed)

    run_dir = args.out_dir / run_id
    common.write_parquet_atomic(results, run_dir / "results.parquet")
    common.write_parquet_atomic(table, run_dir / "metrics.parquet")
    (run_dir / "metrics.md").write_text(fmt_table(table, METRIC_ORDER), encoding="utf-8")

    experiment_id = mlflow_experiment(args.tracking_uri, args.artifact_dir)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_id) as run:
        mlflow.log_params(
            {
                "kind": "pickups",
                "seasons": f"{args.start_season}-{args.end_season}",
                **{k: str(v) for k, v in asdict(config).items()},
                "slots": json.dumps(settings.slots, sort_keys=True),
                "bootstrap": args.bootstrap,
            }
        )
        mlflow.set_tags(
            {
                "git_commit": commit,
                "predictions_path": str(source),
                "predictions_sha256": common.sha256_file(source),
            }
        )
        mlflow.log_metrics(flat_metrics(table))
        mlflow.log_artifact(str(run_dir / "metrics.md"))
        mlflow.log_artifact(str(run_dir / "metrics.parquet"))
    logger.info("pickups %s: %d outcomes, results in %s", run_id, results.height, run_dir)
    logger.info("mlflow run: %s", run.info.run_id)


if __name__ == "__main__":
    main()
