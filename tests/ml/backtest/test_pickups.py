from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest

from ml.backtest import pickups
from ml.backtest.pickups import (
    DEFAULT_COMPOSITION,
    SimConfig,
    rostered_ids,
    sample_rosters,
    simulate,
    summarize,
    wide_predictions,
)
from ml.optimize.league import LeagueSettings

SETTINGS = LeagueSettings()
CONFIG = SimConfig(n_teams=2, rosters_per_week=6, max_moves=(1, 2), seed=3)
COUNTS = {"QB": 8, "RB": 16, "WR": 20, "TE": 8}


def long_predictions(
    seasons: tuple[int, ...] = (2024,), weeks: range = range(2, 5)
) -> pl.DataFrame:
    """A backtest ``predictions.parquet``-shaped frame with known structure."""
    rng = np.random.default_rng(0)
    rows = []
    for season in seasons:
        for week in weeks:
            for pos, count in COUNTS.items():
                for i in range(count):
                    pid = f"{pos}{i}"
                    talent = 20.0 - i
                    label = max(talent + rng.normal(0, 4), 0.0)
                    preds = {
                        "season_avg": talent,
                        "last3": talent + rng.normal(0, 3),
                        "lgbm": label + rng.normal(0, 1),  # informative on purpose
                        "lgbm_kickoff": talent,
                    }
                    for model, pred in preds.items():
                        rows.append(
                            {
                                "player_id": pid,
                                "position": pos,
                                "season": season,
                                "week": week,
                                "label_ppr": label,
                                "model": model,
                                "prediction": pred,
                            }
                        )
    return pl.DataFrame(rows, schema_overrides={"season": pl.Int32, "week": pl.Int32})


@pytest.fixture(scope="module")
def wide() -> pl.DataFrame:
    return wide_predictions(long_predictions(), 2024, 2024)


@pytest.fixture(scope="module")
def results(wide: pl.DataFrame) -> pl.DataFrame:
    return simulate(wide, SETTINGS, CONFIG)


def test_wide_has_one_row_per_player_week(wide: pl.DataFrame) -> None:
    assert wide.height == sum(COUNTS.values()) * 3
    assert {"pred_lgbm", "pred_last3", "pred_season_avg"} <= set(wide.columns)


def test_holdout_season_rejected() -> None:
    with pytest.raises(ValueError, match="holdout"):
        wide_predictions(long_predictions(), 2025, 2026)


def test_missing_model_rejected() -> None:
    preds = long_predictions().filter(pl.col("model") != "last3")
    with pytest.raises(ValueError, match="last3"):
        wide_predictions(preds, 2024, 2024)


def test_ownership_proxy_ignores_labels_and_compared_models(wide: pl.DataFrame) -> None:
    week = wide.filter(pl.col("week") == 2)
    scrambled = week.with_columns(
        pl.col("label_ppr").shuffle(seed=1),
        pl.col("pred_lgbm").shuffle(seed=2),
        pl.col("pred_last3").shuffle(seed=3),
    )
    assert set(rostered_ids(week, 26)) == set(rostered_ids(scrambled, 26))


def test_sampled_rosters_are_reproducible_and_well_formed(wide: pl.DataFrame) -> None:
    pool = wide.filter(pl.col("week") == 2)
    a = sample_rosters(pool, 5, DEFAULT_COMPOSITION, np.random.default_rng(9))
    b = sample_rosters(pool, 5, DEFAULT_COMPOSITION, np.random.default_rng(9))
    assert a == b
    for roster in a:
        assert len({p.player_id for p in roster}) == len(roster) == 13
        for pos, count in DEFAULT_COMPOSITION.items():
            assert sum(p.position == pos for p in roster) == count


def test_sampling_fails_clearly_when_pool_is_too_thin(wide: pl.DataFrame) -> None:
    pool = wide.filter((pl.col("week") == 2) & ~pl.col("player_id").is_in(["QB1", "QB2"]))
    pool = pool.filter((pl.col("position") != "QB") | (pl.col("player_id") == "QB0"))
    with pytest.raises(ValueError, match="QB"):
        sample_rosters(pool, 1, DEFAULT_COMPOSITION, np.random.default_rng(0))


def test_simulation_shape(results: pl.DataFrame) -> None:
    # 3 weeks x 6 rosters x 3 sources x 2 move limits
    assert results.height == 3 * 6 * 3 * 2
    assert (results["moves"] <= results["max_moves"]).all()


def test_simulation_is_reproducible(wide: pl.DataFrame, results: pl.DataFrame) -> None:
    assert simulate(wide, SETTINGS, CONFIG).equals(results)


def test_oracle_never_loses_points_and_bounds_the_models(results: pl.DataFrame) -> None:
    oracle = results.filter(pl.col("source") == "oracle")
    assert (oracle["realized_gain"] >= -1e-9).all()
    assert (oracle["realized_gain"] - oracle["projected_gain"]).abs().max() < 1e-9  # type: ignore[operator]
    keys = ["season", "week", "roster", "max_moves"]
    joined = results.filter(pl.col("source") != "oracle").join(
        oracle.select(*keys, pl.col("realized_after").alias("oracle_after")), on=keys
    )
    # The oracle may stand pat on gains below min_gain, so it can trail by less than that.
    assert (joined["oracle_after"] >= joined["realized_after"] - CONFIG.min_gain - 1e-9).all()


def test_summary_contains_every_metric_and_paired_difference(results: pl.DataFrame) -> None:
    table = summarize(results, n_boot=200)
    for k in (1, 2):
        at_k = table.filter(pl.col("max_moves") == k)
        assert set(at_k["source"]) == {"lgbm", "last3", "oracle", "lgbm - last3"}
        assert (at_k["lo"] <= at_k["point"] + 1e-9).all()
        assert (at_k["point"] <= at_k["hi"] + 1e-9).all()


def test_summary_point_estimates_match_hand_computation() -> None:
    rows = []
    for week, (g_lgbm, g_last3) in enumerate([(2.0, 0.0), (4.0, 1.0)], start=2):
        for source, gain in [("lgbm", g_lgbm), ("last3", g_last3), ("oracle", 5.0)]:
            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "roster": 0,
                    "source": source,
                    "max_moves": 1,
                    "moves": int(gain > 0),
                    "projected_gain": gain,
                    "realized_before": 100.0,
                    "realized_after": 100.0 + gain,
                    "realized_gain": gain,
                }
            )
    table = summarize(pl.DataFrame(rows), n_boot=100)

    def point(source: str, metric: str) -> float:
        row = table.filter((pl.col("source") == source) & (pl.col("metric") == metric))
        return float(row["point"][0])

    assert point("lgbm", "realized_gain") == pytest.approx(3.0)
    assert point("last3", "realized_gain") == pytest.approx(0.5)
    assert point("lgbm - last3", "realized_gain") == pytest.approx(2.5)
    assert point("lgbm - last3", "realized_after") == pytest.approx(2.5)
    assert point("last3", "move_rate") == pytest.approx(0.5)
    assert point("last3", "hit_rate") == pytest.approx(1.0)


def test_cli_writes_results_and_logs_to_mlflow(tmp_path: Path) -> None:
    run_dir = tmp_path / "backtest" / "20240101T000000Z-abc1234"
    run_dir.mkdir(parents=True)
    long_predictions(weeks=range(2, 4)).write_parquet(run_dir / "predictions.parquet")
    pickups.main(
        [
            "--start-season", "2024", "--end-season", "2024",
            "--teams", "2", "--rosters", "2", "--max-moves", "1", "--bootstrap", "50",
            "--out-dir", str(tmp_path / "backtest"),
            "--tracking-uri", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "--artifact-dir", str(tmp_path / "artifacts"),
        ]
    )  # fmt: skip
    (out,) = [p for p in (tmp_path / "backtest").iterdir() if p.name.startswith("pickups-")]
    assert pl.read_parquet(out / "results.parquet").height == 2 * 2 * 3
    assert "lgbm - last3" in (out / "metrics.md").read_text(encoding="utf-8")

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    (run,) = mlflow.search_runs(search_all_experiments=True, output_format="list")
    assert run.data.params["kind"] == "pickups"
    assert run.data.tags["predictions_sha256"]
    assert "k1.lgbm_minus_last3.realized_gain" in run.data.metrics


def test_latest_predictions_skips_pickup_runs(tmp_path: Path) -> None:
    for name in ("20240101T000000Z-aaa", "20240202T000000Z-bbb", "pickups-20240303T000000Z-c"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "predictions.parquet").write_bytes(b"")
    assert pickups.latest_predictions(tmp_path).parent.name == "20240202T000000Z-bbb"


def test_latest_predictions_explains_missing_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"ml\.backtest\.run"):
        pickups.latest_predictions(tmp_path)


def test_cli_rejects_holdout() -> None:
    with pytest.raises(SystemExit):
        pickups.parse_args(["--end-season", "2026"])
