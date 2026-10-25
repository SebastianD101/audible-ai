"""Leakage-free hyperparameter tuning for LightGBM: nested walk-forward by season.

Params for evaluation season S are chosen by fitting on seasons before S-1 and scoring on
season S-1, the same way the backtest refits once per season. Season S is never loaded here, so
the backtest score for S stays out-of-sample. The objective is RMSE on ``relevant`` rows
(``ml.backtest.scopes``), the headline scope. A season without two prior seasons of history
gets the a-priori defaults in ``LgbmParams``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import optuna
import polars as pl

from ml.backtest.scopes import KEYS, relevance
from ml.features.spine import LABEL_COLUMNS
from ml.models.lgbm import LgbmParams, LgbmPredictor, Objective

logger = logging.getLogger(__name__)

DEFAULT_TRIALS = 40
Score = Callable[[LgbmParams, pl.DataFrame, pl.DataFrame], float]


@dataclass(frozen=True)
class Trial:
    """One Optuna trial: its validation RMSE and parameters."""

    number: int
    rmse: float
    params: LgbmParams


@dataclass(frozen=True)
class TuneResult:
    """Chosen params for one evaluation season, and how they were chosen."""

    season: int
    params: LgbmParams
    tuned: bool
    validation_season: int | None
    best_rmse: float | None
    trials: tuple[Trial, ...]


def suggest(trial: optuna.Trial) -> LgbmParams:
    """The search space: small, and stated here rather than in config."""
    objective: Objective = (
        "tweedie"
        if trial.suggest_categorical("objective", ["regression", "tweedie"]) == "tweedie"
        else "regression"
    )
    return LgbmParams(
        objective=objective,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        num_leaves=trial.suggest_int("num_leaves", 8, 64, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 400, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        tweedie_variance_power=trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
        if objective == "tweedie"
        else LgbmParams.tweedie_variance_power,
    )


def relevant_rmse(
    params: LgbmParams, train: pl.DataFrame, valid: pl.DataFrame, *, kickoff: bool = False
) -> float:
    """Fit on ``train``, predict ``valid``, return RMSE over ``valid``'s relevant rows."""
    model = LgbmPredictor(params, kickoff=kickoff)
    model.fit(train)
    scored = valid.with_columns(model.predict(valid.drop(LABEL_COLUMNS)).alias("prediction"))
    rel = scored.join(relevance(valid), on=KEYS, how="left").filter(pl.col("relevant"))
    err = rel["prediction"].to_numpy() - rel["label_ppr"].cast(pl.Float64).to_numpy()
    return float(np.sqrt(np.mean(err**2)))


def tune_season(
    df: pl.DataFrame,
    season: int,
    *,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = 20261025,
    score: Score = relevant_rmse,
) -> TuneResult:
    """Choose params for evaluation season ``season`` using only seasons before it.

    Returns:
        Tuned params, or the defaults (``tuned=False``) if there's no season before S-1.
    """
    history = df.filter(pl.col("season") < season)
    valid_season = season - 1
    train = history.filter(pl.col("season") < valid_season)
    valid = history.filter(pl.col("season") == valid_season)
    if train.is_empty() or valid.is_empty():
        logger.info("season %d: not enough history to tune; using defaults", season)
        return TuneResult(season, LgbmParams(), False, None, None, ())

    trials: list[Trial] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        rmse = score(params, train, valid)
        trials.append(Trial(trial.number, rmse, params))
        return rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = min(trials, key=lambda t: (t.rmse, t.number))
    logger.info(
        "season %d: tuned on %d (train < %d), relevant RMSE %.3f over %d trials",
        season,
        valid_season,
        valid_season,
        best.rmse,
        n_trials,
    )
    return TuneResult(season, best.params, True, valid_season, best.rmse, tuple(trials))


def tune_seasons(
    df: pl.DataFrame,
    seasons: Sequence[int],
    *,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = 20261025,
    score: Score = relevant_rmse,
) -> Mapping[int, TuneResult]:
    """``tune_season`` for each evaluation season."""
    return {s: tune_season(df, s, n_trials=n_trials, seed=seed, score=score) for s in seasons}
