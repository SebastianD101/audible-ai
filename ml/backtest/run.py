"""Run the walk-forward backtest, write results, and log every model to MLflow.

Usage:
    uv run python -m ml.backtest.run --start-season 2019 --end-season 2025

Evaluates each model on every target week of the evaluation seasons, training only on earlier
weeks (2018 serves as burn-in history). Writes::

    ml/data/backtest/<run_id>/predictions.parquet   one row per (model, player-week)
    ml/data/backtest/<run_id>/metrics.parquet       tidy metrics with 95% intervals
    ml/data/backtest/<run_id>/metrics.md            the same as Markdown tables

and one MLflow run per model (params, metrics, git commit, features-manifest hash) in the local
store ``sqlite:///mlruns/mlflow.db``. Browse with
``uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db``.

The 2026 holdout season is rejected and never loaded.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import polars as pl

from ml.backtest.harness import Refit, evaluation_weeks, walk_forward
from ml.backtest.metrics import DEFAULT_BOOTSTRAP, DEFAULT_SEED
from ml.backtest.predictors import BASELINES, Predictor
from ml.backtest.report import metrics_table, to_markdown
from ml.backtest.scopes import attach_scopes
from ml.features.build import DEFAULT_OUT_DIR as FEATURES_DIR
from ml.features.build import git_commit
from ml.ingest import common

logger = logging.getLogger(__name__)

DEFAULT_START_SEASON = 2019
DEFAULT_OUT_DIR = Path("ml/data/backtest")
DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
DEFAULT_ARTIFACT_DIR = Path("mlruns/artifacts")
EXPERIMENT = "audible-ai-backtest"


def load_features(features_dir: Path, end_season: int) -> pl.DataFrame:
    """Read every feature season up to ``end_season`` (never the holdout)."""
    df = pl.scan_parquet(features_dir / "season=*.parquet").filter(
        pl.col("season") <= min(end_season, common.HOLDOUT_SEASON - 1)
    )
    return df.collect()


def mlflow_experiment(tracking_uri: str, artifact_dir: Path) -> str:
    """Point MLflow at the local store and return the experiment ID (created if missing)."""
    mlflow.set_tracking_uri(tracking_uri)
    existing = mlflow.get_experiment_by_name(EXPERIMENT)
    if existing is not None:
        return str(existing.experiment_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return mlflow.create_experiment(EXPERIMENT, artifact_location=artifact_dir.resolve().as_uri())


def flat_metrics(table: pl.DataFrame, model: str) -> dict[str, float]:
    """MLflow metric dict for one model, e.g. ``relevant.WR.rmse`` / ``_lo`` / ``_hi``."""
    out: dict[str, float] = {}
    for r in table.filter(pl.col("model") == model).iter_rows(named=True):
        prefix = "diff_vs_last3." if r["kind"] == "diff" else ""
        key = f"{prefix}{r['scope']}.{r['position']}.{r['metric']}"
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = r["point"], r["lo"], r["hi"]
    return {k: v for k, v in out.items() if v == v}  # drop NaN (e.g. undefined Spearman)


def run_backtest(
    features: pl.DataFrame,
    models: list[Predictor],
    seasons: list[int],
    *,
    refit: Refit,
    n_boot: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (predictions with scopes, metrics table)."""
    weeks = evaluation_weeks(features, seasons)
    preds = attach_scopes(walk_forward(features, models, weeks, refit=refit), features)
    return preds, metrics_table(preds, n_boot=n_boot, seed=seed)


def log_to_mlflow(
    table: pl.DataFrame,
    run_dir: Path,
    *,
    run_id: str,
    params: dict[str, object],
    tags: dict[str, str],
    tracking_uri: str,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> list[str]:
    """Log one MLflow run per model and return the MLflow run IDs."""
    experiment_id = mlflow_experiment(tracking_uri, artifact_dir)
    ids = []
    for model in table["model"].unique(maintain_order=True).to_list():
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model}-{run_id}") as run:
            mlflow.log_params({**params, "model": model})
            mlflow.set_tags({**tags, "backtest_run_id": run_id})
            mlflow.log_metrics(flat_metrics(table, model))
            mlflow.log_artifact(str(run_dir / "metrics.md"))
            mlflow.log_artifact(str(run_dir / "metrics.parquet"))
            ids.append(run.info.run_id)
    return ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments; rejects the holdout season."""
    parser = argparse.ArgumentParser(description=common.first_line(__doc__))
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=common.DEFAULT_END_SEASON)
    parser.add_argument("--models", default=",".join(BASELINES), help="comma-separated")
    parser.add_argument("--refit", choices=["week", "season"], default="week")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = common.parse_season_args(parser, argv)
    if args.end_season >= common.HOLDOUT_SEASON:
        parser.error(f"{common.HOLDOUT_SEASON}+ is the live holdout; backtests stop at 2025")
    unknown = set(args.models.split(",")) - set(BASELINES)
    if unknown:
        parser.error(f"unknown models {sorted(unknown)}; available: {sorted(BASELINES)}")
    return args


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.configure_logging()
    args = parse_args(argv)
    seasons = list(range(args.start_season, args.end_season + 1))
    models = [BASELINES[name]() for name in args.models.split(",")]
    commit = git_commit()
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{commit[:7]}"

    features = load_features(args.features_dir, args.end_season)
    preds, table = run_backtest(
        features, models, seasons, refit=args.refit, n_boot=args.bootstrap, seed=args.seed
    )

    run_dir = args.out_dir / run_id
    common.write_parquet_atomic(preds, run_dir / "predictions.parquet")
    common.write_parquet_atomic(table, run_dir / "metrics.parquet")
    (run_dir / "metrics.md").write_text(to_markdown(table), encoding="utf-8")
    manifest = args.features_dir / "manifest.json"
    ids = log_to_mlflow(
        table,
        run_dir,
        run_id=run_id,
        params={
            "seasons": f"{seasons[0]}-{seasons[-1]}",
            "refit": args.refit,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        tags={
            "git_commit": commit,
            "features_manifest_sha256": common.sha256_file(manifest)
            if manifest.exists()
            else "missing",
        },
        tracking_uri=args.tracking_uri,
        artifact_dir=args.artifact_dir,
    )
    logger.info("backtest %s: %d predictions, results in %s", run_id, preds.height, run_dir)
    logger.info("mlflow runs: %s", ", ".join(ids))


if __name__ == "__main__":
    main()
