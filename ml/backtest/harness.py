"""Walk-forward evaluation: train on the past, predict one target week, step forward.

To predict week W, a model may train only on rows whose *target* week is before W. Those rows'
labels are final by Tuesday of week W, so this matches the waiver-time cutoff the features use.
Week-W rows reach ``predict`` with every label column removed. Random train/test splits are
never used: they would let a model learn from the future of the same season.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import polars as pl

from ml.backtest.predictors import Predictor
from ml.features.asof import Week, available_before
from ml.features.spine import LABEL_COLUMNS, SPINE_COLUMNS

Refit = Literal["week", "season"]


def evaluation_weeks(df: pl.DataFrame, seasons: Sequence[int]) -> list[Week]:
    """Every target week present in ``df`` for ``seasons``, in time order."""
    pairs = (
        df.filter(pl.col("season").is_in(list(seasons)))
        .select("season", "week")
        .unique()
        .sort("season", "week")
        .rows()
    )
    return [Week(s, w) for s, w in pairs]


def week_rows(df: pl.DataFrame, week: Week) -> pl.DataFrame:
    """Rows for exactly one target week."""
    return df.filter(pl.col("season") == week.season, pl.col("week") == week.week)


def check_predictions(pred: pl.Series, expected_len: int, model: str, week: Week) -> pl.Series:
    """Require one finite prediction per row.

    Raises:
        ValueError: If the length is wrong or any prediction is null, NaN, or infinite.
    """
    if pred.len() != expected_len:
        raise ValueError(f"{model} {week}: {pred.len()} predictions for {expected_len} rows")
    pred = pred.cast(pl.Float64)
    if pred.null_count() or not pred.is_finite().all():
        raise ValueError(f"{model} {week}: null or non-finite predictions")
    return pred


def walk_forward(
    df: pl.DataFrame,
    predictors: Sequence[Predictor],
    weeks: Sequence[Week],
    *,
    refit: Refit = "week",
) -> pl.DataFrame:
    """Evaluate each predictor over ``weeks`` and return one row per (model, player-week).

    With ``refit="season"`` a model is fit once per season on everything before that season's
    first test week, which is still strictly in the past for every later week of the season.

    Returns:
        Identifier and label columns plus ``model`` and ``prediction``.
    """
    results = []
    for model in predictors:
        fitted_for: int | None = None
        for week in weeks:
            if refit == "week" or fitted_for != week.season:
                model.fit(available_before(df, week))
                fitted_for = week.season
            test = week_rows(df, week)
            if test.is_empty():
                continue
            pred = model.predict(test.drop(LABEL_COLUMNS))
            results.append(
                test.select(*SPINE_COLUMNS, *LABEL_COLUMNS).with_columns(
                    pl.lit(model.name).alias("model"),
                    check_predictions(pred, test.height, model.name, week).alias("prediction"),
                )
            )
    if not results:
        raise ValueError("no test rows for the requested weeks")
    return pl.concat(results)
