"""The interface every projection model implements, and the naive baselines.

A predictor is fit on past rows (features and labels) and then predicts PPR points for one target
week from features alone; the harness strips label columns before calling ``predict``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import polars as pl


class Predictor(Protocol):
    """A projection model the walk-forward harness can fit and evaluate."""

    @property
    def name(self) -> str:
        """Short identifier used in results and MLflow."""
        ...

    def fit(self, train: pl.DataFrame) -> None:
        """Learn from rows whose target week precedes the evaluation week."""
        ...

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Return projected PPR points, one per row of ``features``."""
        ...


def with_fallbacks(features: pl.DataFrame, primary: str) -> pl.Series:
    """``primary`` per-game PPR, else prior-season PPG, else 0."""
    return features.select(
        pl.coalesce(pl.col(primary), pl.col("prev_ppr"), pl.lit(0.0)).alias("prediction")
    ).to_series()


class Last3Baseline:
    """Mean PPR over the last 3 games played (the standard naive projection).

    Ignores availability: a player who just got hurt is projected as if healthy.
    """

    name = "last3"

    def fit(self, train: pl.DataFrame) -> None:
        """No-op: the baseline has no parameters."""

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Return ``ppr_last3`` with prior-season and zero fallbacks."""
        return with_fallbacks(features, "ppr_last3")


class SeasonAvgBaseline:
    """Season-to-date PPR per game played."""

    name = "season_avg"

    def fit(self, train: pl.DataFrame) -> None:
        """No-op: the baseline has no parameters."""

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Return ``ppr_season`` with prior-season and zero fallbacks."""
        return with_fallbacks(features, "ppr_season")


BASELINES: dict[str, Callable[[], Predictor]] = {
    Last3Baseline.name: Last3Baseline,
    SeasonAvgBaseline.name: SeasonAvgBaseline,
}
