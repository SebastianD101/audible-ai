import math

import polars as pl
import pytest

from ml.backtest import metrics


def preds(rows: list[tuple[int, str, float, float]], season: int = 2022) -> pl.DataFrame:
    """rows: (week, position, prediction, label)."""
    return pl.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(len(rows))],
            "season": [season] * len(rows),
            "week": [r[0] for r in rows],
            "position": [r[1] for r in rows],
            "prediction": [r[2] for r in rows],
            "label_ppr": [r[3] for r in rows],
        }
    )


def test_rmse_and_mae_match_hand_computation() -> None:
    df = preds([(2, "WR", 1.0, 1.0), (2, "WR", 2.0, 4.0), (2, "WR", 3.0, 0.0)])
    est = metrics.bootstrap(df, n_boot=10)
    assert est["rmse"].point == pytest.approx(math.sqrt(13 / 3))
    assert est["mae"].point == pytest.approx(5 / 3)
    assert (est["rmse"].n_weeks, est["rmse"].n_rows) == (1, 3)


def test_spearman_is_within_week_and_position_not_pooled() -> None:
    # Perfect ordering inside each week, but week 3's predictions are all higher while its
    # labels are all lower, so a pooled correlation would be poor.
    df = preds(
        [
            *[(2, "WR", 1.0, 10.0), (2, "WR", 2.0, 20.0), (2, "WR", 3.0, 30.0)],
            *[(3, "WR", 4.0, 1.0), (3, "WR", 5.0, 2.0), (3, "WR", 6.0, 3.0)],
        ]
    )
    assert metrics.bootstrap(df, n_boot=10)["spearman"].point == pytest.approx(1.0)
    pooled = df.select(pl.corr("prediction", "label_ppr", method="spearman")).item()
    assert pooled < 0.5


def test_spearman_skips_groups_too_small_to_rank() -> None:
    df = preds(
        [
            *[(2, "TE", 1.0, 5.0), (2, "TE", 2.0, 1.0)],
            *[(2, "QB", 1.0, 1.0), (2, "QB", 2.0, 2.0), (2, "QB", 3.0, 3.0)],
        ]
    )
    stats = metrics.week_stats(df)
    assert stats["rho_n"].item() == 1
    assert metrics.bootstrap(df, n_boot=10)["spearman"].point == pytest.approx(1.0)


def varied_weeks(n_weeks: int = 12) -> pl.DataFrame:
    rows = []
    for w in range(2, 2 + n_weeks):
        miss = float(w % 4)  # weeks differ in how wrong the model is
        rows += [(w, "RB", 10.0, 10.0 + miss), (w, "RB", 5.0, 5.0 - miss), (w, "RB", 1.0, 2.0)]
    return preds(rows)


def test_bootstrap_is_deterministic_for_a_seed_and_contains_the_point() -> None:
    a = metrics.bootstrap(varied_weeks(), n_boot=200, seed=7)
    b = metrics.bootstrap(varied_weeks(), n_boot=200, seed=7)
    c = metrics.bootstrap(varied_weeks(), n_boot=200, seed=8)
    assert a == b
    assert a["rmse"] != c["rmse"]
    for est in a.values():
        assert est.lo <= est.point <= est.hi


def test_bootstrap_resamples_weeks_not_rows() -> None:
    # Every week has identical errors, so resampling whole weeks cannot move the metric,
    # while resampling individual rows would.
    rows = [r for w in range(2, 12) for r in [(w, "WR", 0.0, 0.0), (w, "WR", 0.0, 2.0)]]
    est = metrics.bootstrap(preds(rows), n_boot=200)
    assert est["rmse"].lo == pytest.approx(est["rmse"].point)
    assert est["rmse"].hi == pytest.approx(est["rmse"].point)


def test_paired_difference_of_a_model_with_itself_is_zero() -> None:
    df = varied_weeks()
    diff = metrics.paired_bootstrap(df, df, n_boot=100)
    for est in diff.values():
        assert (est.point, est.lo, est.hi) == (0.0, 0.0, 0.0)


def test_paired_difference_detects_a_uniformly_better_model() -> None:
    worse = varied_weeks()
    better = worse.with_columns(
        ((pl.col("prediction") + pl.col("label_ppr")) / 2).alias("prediction")
    )
    diff = metrics.paired_bootstrap(better, worse, n_boot=200)
    assert diff["rmse"].hi < 0
    assert diff["mae"].hi < 0


def test_paired_bootstrap_requires_same_rows() -> None:
    df = varied_weeks()
    with pytest.raises(ValueError, match="same player-weeks"):
        metrics.paired_bootstrap(df, df.head(5), n_boot=10)
