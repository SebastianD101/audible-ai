import polars as pl
import pytest

from ml.backtest import harness
from ml.backtest.predictors import Last3Baseline
from ml.features.asof import Week
from ml.features.spine import LABEL_COLUMNS
from tests.ml.backtest.frames import feature_table


class Spy:
    """Records every frame the harness hands to fit and predict."""

    name = "spy"

    def __init__(self) -> None:
        self.fit_calls: list[pl.DataFrame] = []
        self.predict_calls: list[pl.DataFrame] = []

    def fit(self, train: pl.DataFrame) -> None:
        self.fit_calls.append(train)

    def predict(self, features: pl.DataFrame) -> pl.Series:
        self.predict_calls.append(features)
        return pl.Series([1.0] * features.height)


def latest(df: pl.DataFrame) -> tuple[int, int] | None:
    if df.is_empty():
        return None
    return max(df.select("season", "week").rows())


def test_evaluation_weeks_are_ordered_and_limited_to_seasons() -> None:
    weeks = harness.evaluation_weeks(feature_table(seasons=(2020, 2021, 2022)), [2022, 2021])
    assert weeks[0] == Week(2021, 2)
    assert weeks[-1] == Week(2022, 4)
    assert len(weeks) == 6


def test_weekly_refit_trains_only_on_earlier_target_weeks() -> None:
    df = feature_table()
    spy = Spy()
    weeks = harness.evaluation_weeks(df, [2022])
    harness.walk_forward(df, [spy], weeks)
    assert len(spy.fit_calls) == len(weeks)
    for week, train in zip(weeks, spy.fit_calls, strict=True):
        newest = latest(train)
        assert newest is not None
        assert newest < (week.season, week.week)
    # The last fit sees everything before 2022 week 4, including 2022 weeks 2-3.
    assert latest(spy.fit_calls[-1]) == (2022, 3)


def test_season_refit_fits_once_per_season_before_its_first_test_week() -> None:
    df = feature_table(seasons=(2020, 2021, 2022))
    spy = Spy()
    harness.walk_forward(df, [spy], harness.evaluation_weeks(df, [2021, 2022]), refit="season")
    assert [latest(t) for t in spy.fit_calls] == [(2020, 4), (2021, 4)]


def test_predict_never_sees_labels_or_other_weeks() -> None:
    df = feature_table()
    spy = Spy()
    weeks = harness.evaluation_weeks(df, [2022])
    harness.walk_forward(df, [spy], weeks)
    for week, features in zip(weeks, spy.predict_calls, strict=True):
        assert not set(LABEL_COLUMNS) & set(features.columns)
        assert features.select("season", "week").unique().rows() == [(week.season, week.week)]


def test_output_has_one_row_per_model_and_player_week() -> None:
    df = feature_table()
    weeks = harness.evaluation_weeks(df, [2022])
    out = harness.walk_forward(df, [Last3Baseline(), Spy()], weeks)
    assert out.height == 2 * df.filter(pl.col("season") == 2022).height
    assert set(out["model"]) == {"last3", "spy"}
    assert {"prediction", "label_ppr"} <= set(out.columns)


class Broken:
    name = "broken"

    def __init__(self, values: list[float | None]) -> None:
        self.values = values

    def fit(self, train: pl.DataFrame) -> None:
        pass

    def predict(self, features: pl.DataFrame) -> pl.Series:
        return pl.Series(self.values, dtype=pl.Float64)


@pytest.mark.parametrize(
    ("values", "message"),
    [([1.0], "predictions for"), ([None] * 12, "non-finite"), ([float("nan")] * 12, "non-finite")],
)
def test_bad_predictions_are_rejected(values: list[float | None], message: str) -> None:
    df = feature_table()
    with pytest.raises(ValueError, match=message):
        harness.walk_forward(df, [Broken(values)], [Week(2022, 2)])
