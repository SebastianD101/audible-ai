import polars as pl

from ml.models import tune
from ml.models.lgbm import LgbmParams
from tests.ml.backtest.frames import feature_table


class SpyScore:
    """Records the seasons each trial trains and validates on; prefers fewer trees."""

    def __init__(self) -> None:
        self.calls: list[tuple[set[int], set[int]]] = []

    def __call__(self, params: LgbmParams, train: pl.DataFrame, valid: pl.DataFrame) -> float:
        self.calls.append((set(train["season"]), set(valid["season"])))
        return float(params.n_estimators)


def test_tuning_for_a_season_never_sees_that_season_or_later() -> None:
    df = feature_table(seasons=(2018, 2019, 2020, 2021, 2022))
    spy = SpyScore()
    result = tune.tune_season(df, 2021, n_trials=5, score=spy)
    assert len(spy.calls) == 5
    for train_seasons, valid_seasons in spy.calls:
        assert train_seasons == {2018, 2019}
        assert valid_seasons == {2020}
    assert result.tuned
    assert result.validation_season == 2020
    assert result.params.n_estimators == min(t.params.n_estimators for t in result.trials)


def test_season_without_two_prior_seasons_uses_defaults() -> None:
    df = feature_table(seasons=(2018, 2019))
    spy = SpyScore()
    result = tune.tune_season(df, 2019, n_trials=3, score=spy)
    assert not result.tuned
    assert result.params == LgbmParams()
    assert spy.calls == []


def test_tuning_is_reproducible_with_a_seed() -> None:
    df = feature_table(seasons=(2018, 2019, 2020))
    a = tune.tune_season(df, 2020, n_trials=4, score=SpyScore())
    b = tune.tune_season(df, 2020, n_trials=4, score=SpyScore())
    assert [t.params for t in a.trials] == [t.params for t in b.trials]


def test_relevant_rmse_runs_a_real_fit() -> None:
    df = feature_table(seasons=(2019, 2020), weeks=range(2, 8), per_position=4)
    small = LgbmParams(n_estimators=10, min_data_in_leaf=2, num_leaves=4)
    rmse = tune.relevant_rmse(
        small, df.filter(pl.col("season") == 2019), df.filter(pl.col("season") == 2020)
    )
    assert 0 <= rmse < 20


def test_tune_seasons_covers_each_season() -> None:
    df = feature_table(seasons=(2018, 2019, 2020))
    out = tune.tune_seasons(df, [2019, 2020], n_trials=2, score=SpyScore())
    assert set(out) == {2019, 2020}
    assert not out[2019].tuned
    assert out[2020].tuned
