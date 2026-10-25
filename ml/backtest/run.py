"""Run the walk-forward backtest, write results, and log every model to MLflow.

Usage:
    uv run python -m ml.backtest.run --start-season 2019 --end-season 2025

Evaluates each model on every target week of the evaluation seasons, training only on earlier
weeks (2018 serves as burn-in history). Writes::

    ml/data/backtest/<run_id>/predictions.parquet   one row per (model, player-week)
    ml/data/backtest/<run_id>/metrics.parquet       tidy metrics with 95% intervals
    ml/data/backtest/<run_id>/metrics.md            the same as Markdown tables
    ml/data/backtest/<run_id>/importance.parquet    LightGBM gain per feature (if run)

and one MLflow run per model (params, metrics, git commit, features-manifest hash) in the local
store ``sqlite:///mlruns/mlflow.db``. Browse with
``uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db``.

LightGBM models (``lgbm``, and the ``lgbm_kickoff`` ablation) are tuned per evaluation season by
nested walk-forward (``ml.models.tune``) unless ``--no-tune`` is given. Each season's choice is
logged as an MLflow child run.

The 2026 holdout season is rejected and never loaded.
"""

from __future__ import annotations

import argparse
import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
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
from ml.models.lgbm import LgbmParams, LgbmPredictor
from ml.models.tune import DEFAULT_TRIALS, TuneResult, relevant_rmse, tune_seasons

logger = logging.getLogger(__name__)

DEFAULT_START_SEASON = 2019
DEFAULT_OUT_DIR = Path("ml/data/backtest")
DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
DEFAULT_ARTIFACT_DIR = Path("mlruns/artifacts")
EXPERIMENT = "audible-ai-backtest"
LGBM_MODELS = {"lgbm": False, "lgbm_kickoff": True}  # name -> uses kickoff columns
MODEL_NAMES = [*BASELINES, *LGBM_MODELS]
DEFAULT_MODELS = "last3,season_avg,lgbm"

Tuning = Mapping[str, Mapping[int, TuneResult]]  # model -> evaluation season -> result


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
    """MLflow metric dict for one model, e.g. ``relevant.WR.rmse`` / ``_lo`` / ``_hi``.

    Diffs are keyed ``diff_vs_<reference>.<scope>.<position>.<metric>``.
    """
    out: dict[str, float] = {}
    for r in table.filter(pl.col("model") == model).iter_rows(named=True):
        prefix = f"diff_vs_{r['reference']}." if r["kind"] == "diff" else ""
        key = f"{prefix}{r['scope']}.{r['position']}.{r['metric']}"
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = r["point"], r["lo"], r["hi"]
    return {k: v for k, v in out.items() if v == v}  # drop NaN (e.g. undefined Spearman)


def tune_models(
    features: pl.DataFrame,
    names: Sequence[str],
    seasons: Sequence[int],
    *,
    tune: bool,
    n_trials: int,
    seed: int,
) -> dict[str, dict[int, TuneResult]]:
    """Per-season params for each requested LightGBM model (defaults when ``tune`` is off)."""
    out: dict[str, dict[int, TuneResult]] = {}
    for name in (n for n in names if n in LGBM_MODELS):
        if not tune:
            out[name] = {s: TuneResult(s, LgbmParams(), False, None, None, ()) for s in seasons}
            continue
        score = functools.partial(relevant_rmse, kickoff=LGBM_MODELS[name])
        out[name] = dict(tune_seasons(features, seasons, n_trials=n_trials, seed=seed, score=score))
    return out


def build_models(names: Sequence[str], season: int, tuning: Tuning) -> list[Predictor]:
    """Fresh predictors for one evaluation season, LightGBM with that season's params."""
    return [
        LgbmPredictor(tuning[n][season].params, kickoff=LGBM_MODELS[n])
        if n in LGBM_MODELS
        else BASELINES[n]()
        for n in names
    ]


def run_backtest(
    features: pl.DataFrame,
    models_for_season: Callable[[int], list[Predictor]],
    seasons: list[int],
    *,
    refit: Refit,
    n_boot: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return (predictions with scopes, metrics table, LightGBM gain importance).

    Models are rebuilt each season so tuned params can differ by season. Within a season the
    harness refits as ``refit`` says. Importance comes from each season's final fit.
    """
    parts, importance = [], []
    for season in seasons:
        weeks = evaluation_weeks(features, [season])
        if not weeks:
            continue
        models = models_for_season(season)
        parts.append(walk_forward(features, models, weeks, refit=refit))
        importance += [
            m.importance().with_columns(model=pl.lit(m.name), season=pl.lit(season))
            for m in models
            if isinstance(m, LgbmPredictor)
        ]
    if not parts:
        raise ValueError(f"no test rows for seasons {seasons}")
    preds = attach_scopes(pl.concat(parts), features)
    imp = pl.concat(importance) if importance else pl.DataFrame()
    return preds, metrics_table(preds, n_boot=n_boot, seed=seed), imp


def log_to_mlflow(
    table: pl.DataFrame,
    run_dir: Path,
    *,
    run_id: str,
    params: dict[str, object],
    tags: dict[str, str],
    tracking_uri: str,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    tuning: Tuning | None = None,
) -> list[str]:
    """Log one MLflow run per model and return the MLflow run IDs.

    LightGBM models get one child run per evaluation season recording the chosen params.
    """
    experiment_id = mlflow_experiment(tracking_uri, artifact_dir)
    ids = []
    for model in table["model"].unique(maintain_order=True).to_list():
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model}-{run_id}") as run:
            mlflow.log_params({**params, "model": model})
            mlflow.set_tags({**tags, "backtest_run_id": run_id})
            mlflow.log_metrics(flat_metrics(table, model))
            mlflow.log_artifact(str(run_dir / "metrics.md"))
            mlflow.log_artifact(str(run_dir / "metrics.parquet"))
            if model in LGBM_MODELS and (run_dir / "importance.parquet").exists():
                mlflow.log_artifact(str(run_dir / "importance.parquet"))
            for _, result in sorted((tuning or {}).get(model, {}).items()):
                log_tuning(experiment_id, f"{model}-{result.season}-{run_id}", result)
            ids.append(run.info.run_id)
    return ids


def log_tuning(experiment_id: str, run_name: str, result: TuneResult) -> None:
    """Child run recording how one evaluation season's params were chosen."""
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        mlflow.log_params(
            {
                **asdict(result.params),
                "evaluation_season": result.season,
                "validation_season": result.validation_season,
                "tuned": result.tuned,
                "trials": len(result.trials),
            }
        )
        if result.best_rmse is not None:
            mlflow.log_metric("validation_relevant_rmse", result.best_rmse)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments; rejects the holdout season."""
    parser = argparse.ArgumentParser(description=common.first_line(__doc__))
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=common.DEFAULT_END_SEASON)
    parser.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated")
    parser.add_argument("--refit", choices=["week", "season"], default="week")
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="tune LightGBM per season by nested walk-forward; --no-tune uses defaults",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = common.parse_season_args(parser, argv)
    if args.end_season >= common.HOLDOUT_SEASON:
        parser.error(f"{common.HOLDOUT_SEASON}+ is the live holdout; backtests stop at 2025")
    unknown = set(args.models.split(",")) - set(MODEL_NAMES)
    if unknown:
        parser.error(f"unknown models {sorted(unknown)}; available: {MODEL_NAMES}")
    return args


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.configure_logging()
    args = parse_args(argv)
    seasons = list(range(args.start_season, args.end_season + 1))
    names = args.models.split(",")
    commit = git_commit()
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{commit[:7]}"

    features = load_features(args.features_dir, args.end_season)
    tuning = tune_models(
        features, names, seasons, tune=args.tune, n_trials=args.trials, seed=args.seed
    )
    preds, table, importance = run_backtest(
        features,
        lambda season: build_models(names, season, tuning),
        seasons,
        refit=args.refit,
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    run_dir = args.out_dir / run_id
    common.write_parquet_atomic(preds, run_dir / "predictions.parquet")
    common.write_parquet_atomic(table, run_dir / "metrics.parquet")
    if not importance.is_empty():
        common.write_parquet_atomic(importance, run_dir / "importance.parquet")
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
            "tune": args.tune,
            "trials": args.trials,
        },
        tags={
            "git_commit": commit,
            "features_manifest_sha256": common.sha256_file(manifest)
            if manifest.exists()
            else "missing",
        },
        tracking_uri=args.tracking_uri,
        artifact_dir=args.artifact_dir,
        tuning=tuning,
    )
    logger.info("backtest %s: %d predictions, results in %s", run_id, preds.height, run_dir)
    logger.info("mlflow runs: %s", ", ".join(ids))


if __name__ == "__main__":
    main()
