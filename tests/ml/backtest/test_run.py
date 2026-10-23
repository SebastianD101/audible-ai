from pathlib import Path

import polars as pl
import pytest
from mlflow.tracking import MlflowClient

from ml.backtest import run
from tests.ml.backtest.frames import feature_table


def write_features(features_dir: Path, df: pl.DataFrame) -> None:
    features_dir.mkdir(parents=True)
    for (season,), part in df.partition_by("season", as_dict=True).items():
        part.write_parquet(features_dir / f"season={season}.parquet")
    (features_dir / "manifest.json").write_text("{}", encoding="utf-8")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--end-season", "2026"], "holdout"),
        (["--start-season", "2026", "--end-season", "2026"], "holdout"),
        (["--models", "last3,crystal_ball"], "unknown models"),
    ],
)
def test_parse_args_rejects_holdout_and_unknown_models(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        run.parse_args(argv)
    assert message in capsys.readouterr().err


def test_parse_args_defaults() -> None:
    args = run.parse_args([])
    assert (args.start_season, args.end_season, args.refit) == (2019, 2025, "week")
    assert args.models == "last3,season_avg"


def test_load_features_never_returns_holdout_rows(tmp_path: Path) -> None:
    write_features(tmp_path / "f", feature_table(seasons=(2025, 2026)))
    assert set(run.load_features(tmp_path / "f", 2030)["season"]) == {2025}


def test_flat_metrics_names_and_drops_nan() -> None:
    table = pl.DataFrame(
        {
            "model": ["season_avg"] * 2,
            "scope": ["relevant"] * 2,
            "position": ["WR", "TE"],
            "kind": ["diff", "value"],
            "metric": ["rmse", "spearman"],
            "point": [-0.5, float("nan")],
            "lo": [-0.9, float("nan")],
            "hi": [-0.1, float("nan")],
        }
    )
    assert run.flat_metrics(table, "season_avg") == {
        "diff_vs_last3.relevant.WR.rmse": -0.5,
        "diff_vs_last3.relevant.WR.rmse_lo": -0.9,
        "diff_vs_last3.relevant.WR.rmse_hi": -0.1,
    }


def test_main_writes_results_and_logs_each_model_to_mlflow(tmp_path: Path) -> None:
    write_features(tmp_path / "features", feature_table(seasons=(2021, 2022)))
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    run.main(
        [
            "--start-season", "2022",
            "--end-season", "2022",
            "--features-dir", str(tmp_path / "features"),
            "--out-dir", str(tmp_path / "out"),
            "--tracking-uri", tracking_uri,
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--bootstrap", "20",
        ]
    )  # fmt: skip

    [run_dir] = list((tmp_path / "out").iterdir())
    assert {p.name for p in run_dir.iterdir()} == {
        "predictions.parquet",
        "metrics.parquet",
        "metrics.md",
    }
    assert pl.read_parquet(run_dir / "predictions.parquet").height == 2 * 3 * 12

    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(run.EXPERIMENT)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert {r.data.params["model"] for r in runs} == {"last3", "season_avg"}
    for r in runs:
        assert r.data.params["seasons"] == "2022-2022"
        assert r.data.tags["git_commit"]
        assert r.data.tags["features_manifest_sha256"] != "missing"
        assert "relevant.ALL.rmse" in r.data.metrics
