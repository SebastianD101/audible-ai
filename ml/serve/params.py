"""Choose the LightGBM params used for live projections of a season, and save them as JSON.

Usage:
    uv run python -m ml.serve.params --season 2026

Uses the same nested walk-forward as the backtest (``ml.models.tune.tune_season``): fit on
seasons before S-1, score on S-1. For the 2026 live season that means train 2018-2024 and
validate on 2025, so 2026 is never used for tuning. The result is written to
``ml/serve/params/season=YYYY.json``, which is committed so the serving params are reviewable,
and logged to MLflow as a child-style tuning run.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from ml.backtest.run import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_TRACKING_URI,
    load_features,
    log_tuning,
    mlflow_experiment,
)
from ml.features.build import DEFAULT_OUT_DIR as FEATURES_DIR
from ml.features.build import git_commit
from ml.ingest import common
from ml.models.lgbm import LgbmParams
from ml.models.tune import DEFAULT_TRIALS, TuneResult, tune_season

logger = logging.getLogger(__name__)

DEFAULT_PARAMS_DIR = Path("ml/serve/params")
DEFAULT_SEED = 20261025


def params_path(season: int, params_dir: Path = DEFAULT_PARAMS_DIR) -> Path:
    """Where the serving params for ``season`` live."""
    return params_dir / f"season={season}.json"


def params_record(result: TuneResult, *, commit: str, features_sha256: str) -> dict[str, Any]:
    """JSON-ready record of a tuning result and where it came from."""
    return {
        "season": result.season,
        "params": asdict(result.params),
        "tuned": result.tuned,
        "validation_season": result.validation_season,
        "validation_relevant_rmse": result.best_rmse,
        "trials": len(result.trials),
        "git_commit": commit,
        "features_manifest_sha256": features_sha256,
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def read_params(path: Path) -> LgbmParams:
    """Load serving params written by this module.

    Raises:
        FileNotFoundError: with the command that creates the file.
        ValueError: if the file has unknown or missing parameter names.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; create it with `uv run python -m ml.serve.params --season ...`"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))["params"]
    expected = {f.name for f in fields(LgbmParams)}
    if set(raw) != expected:
        raise ValueError(f"{path}: params keys {sorted(raw)} != {sorted(expected)}")
    return LgbmParams(**raw)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=common.first_line(__doc__))
    parser.add_argument("--season", type=int, default=common.HOLDOUT_SEASON)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--params-dir", type=Path, default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args(argv)
    if args.season < common.HOLDOUT_SEASON:
        parser.error("serving params are for live seasons; past seasons are tuned by the backtest")
    return args


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.configure_logging()
    args = parse_args(argv)
    # load_features never returns the holdout season, so tuning can't see it.
    features = load_features(args.features_dir, args.season - 1)
    result = tune_season(features, args.season, n_trials=args.trials, seed=args.seed)
    manifest = args.features_dir / "manifest.json"
    commit = git_commit()
    record = params_record(
        result,
        commit=commit,
        features_sha256=common.sha256_file(manifest) if manifest.exists() else "missing",
    )
    path = params_path(args.season, args.params_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    experiment_id = mlflow_experiment(args.tracking_uri, args.artifact_dir)
    run_name = f"serve-params-{args.season}-{commit[:7]}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.set_tags({"git_commit": commit, "kind": "serve-params"})
        log_tuning(experiment_id, f"lgbm-{args.season}-{commit[:7]}", result)
    logger.info("serving params for %d written to %s", args.season, path)


if __name__ == "__main__":
    main()
