import json
from pathlib import Path

import polars as pl
import pytest
from mlflow.tracking import MlflowClient

from ml.backtest.run import EXPERIMENT
from ml.models.lgbm import LgbmParams
from ml.serve import params
from tests.ml.backtest.frames import feature_table


def write_features(features_dir: Path, df: pl.DataFrame) -> None:
    features_dir.mkdir(parents=True)
    for (season,), part in df.group_by("season"):
        part.write_parquet(features_dir / f"season={season}.parquet")
    (features_dir / "manifest.json").write_text("{}", encoding="utf-8")


def test_main_tunes_on_the_prior_season_and_writes_reviewable_json(tmp_path: Path) -> None:
    # A 2026 file is present on purpose: tuning must not read it.
    write_features(tmp_path / "features", feature_table(seasons=(2024, 2025, 2026)))
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    params.main(
        [
            "--season", "2026", "--trials", "2",
            "--features-dir", str(tmp_path / "features"),
            "--params-dir", str(tmp_path / "params"),
            "--tracking-uri", tracking_uri,
            "--artifact-dir", str(tmp_path / "artifacts"),
        ]
    )  # fmt: skip
    path = params.params_path(2026, tmp_path / "params")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["season"] == 2026
    assert record["validation_season"] == 2025
    assert record["tuned"] is True
    assert record["trials"] == 2
    assert record["features_manifest_sha256"] != "missing"
    assert isinstance(params.read_params(path), LgbmParams)

    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert {r.data.tags.get("kind") for r in runs} >= {"serve-params"}
    [child] = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert child.data.params["evaluation_season"] == "2026"


def test_read_params_explains_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"ml\.serve\.params"):
        params.read_params(tmp_path / "season=2026.json")


def test_read_params_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "season=2026.json"
    path.write_text(json.dumps({"params": {"learning_rate": 0.1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="params keys"):
        params.read_params(path)


def test_past_seasons_rejected() -> None:
    with pytest.raises(SystemExit):
        params.parse_args(["--season", "2025"])
