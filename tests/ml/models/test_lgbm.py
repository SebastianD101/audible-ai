import numpy as np
import polars as pl
import pytest

from ml.backtest import harness
from ml.features.spine import LABEL_COLUMNS
from ml.models.lgbm import LgbmParams, LgbmPredictor
from tests.ml.backtest.frames import feature_table

SMALL = LgbmParams(n_estimators=20, min_data_in_leaf=2, num_leaves=4)


def noisy_table() -> pl.DataFrame:
    df = feature_table(seasons=(2020, 2021, 2022), weeks=range(2, 8), per_position=4)
    rng = np.random.default_rng(0)
    return df.with_columns(
        pl.Series("rest", rng.integers(6, 14, df.height)),
        pl.Series("ko_total_line", rng.normal(45, 3, df.height)),
    )


def test_fit_predict_one_finite_nonnegative_value_per_row() -> None:
    df = noisy_table()
    model = LgbmPredictor(SMALL)
    model.fit(df.filter(pl.col("season") < 2022))
    pred = model.predict(df.filter(pl.col("season") == 2022).drop(LABEL_COLUMNS))
    assert pred.len() == df.filter(pl.col("season") == 2022).height
    assert pred.is_finite().all()
    assert pred.min() >= 0  # type: ignore[operator]


def test_learns_the_signal() -> None:
    df = noisy_table()
    model = LgbmPredictor(SMALL)
    model.fit(df.filter(pl.col("season") < 2022))
    test = df.filter(pl.col("season") == 2022)
    pred = model.predict(test.drop(LABEL_COLUMNS))
    assert np.corrcoef(pred.to_numpy(), test["label_ppr"].to_numpy())[0, 1] > 0.9


def test_refits_are_deterministic() -> None:
    df = noisy_table()
    preds = []
    for _ in range(2):
        model = LgbmPredictor(LgbmParams(n_estimators=20, min_data_in_leaf=2, objective="tweedie"))
        model.fit(df.filter(pl.col("season") < 2022))
        preds.append(model.predict(df.filter(pl.col("season") == 2022)))
    assert preds[0].equals(preds[1])


def test_kickoff_columns_only_in_the_ablation() -> None:
    df = noisy_table()
    default, ablation = LgbmPredictor(SMALL), LgbmPredictor(SMALL, kickoff=True)
    for model in (default, ablation):
        model.fit(df)
    assert "ko_total_line" not in default.columns
    assert "ko_total_line" in ablation.columns
    assert (default.name, ablation.name) == ("lgbm", "lgbm_kickoff")
    assert not set(LABEL_COLUMNS) & set(ablation.columns)


def test_runs_in_the_walk_forward_harness() -> None:
    df = noisy_table()
    weeks = harness.evaluation_weeks(df, [2022])
    out = harness.walk_forward(df, [LgbmPredictor(SMALL)], weeks, refit="season")
    assert out.height == df.filter(pl.col("season") == 2022).height


def test_importance_lists_every_feature() -> None:
    model = LgbmPredictor(SMALL)
    model.fit(noisy_table())
    imp = model.importance()
    assert set(imp["feature"]) == set(model.columns)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        LgbmPredictor().predict(feature_table())
