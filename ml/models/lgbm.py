"""Gradient-boosted projection of next-week PPR points (LightGBM).

One model for all positions, with ``position`` as a categorical: it gets about four times the
training rows of a per-position model, and the trees still split on position wherever usage
patterns differ. Trained on every spine row, zeros included, because projecting that a player
won't score (sits, inactive, hurt) is part of the job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import lightgbm as lgb
import numpy as np
import polars as pl

from ml.models.matrix import categorical_indices, feature_columns, to_matrix

LABEL = "label_ppr"
Objective = Literal["regression", "tweedie"]


@dataclass(frozen=True)
class LgbmParams:
    """Hyperparameters. Defaults are a-priori choices, used wherever no tuned set exists."""

    objective: Objective = "regression"
    learning_rate: float = 0.05
    n_estimators: int = 400
    num_leaves: int = 31
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    tweedie_variance_power: float = 1.5

    def booster_params(self, seed: int, num_threads: int) -> dict[str, object]:
        """LightGBM ``params`` dict, with determinism settings."""
        params: dict[str, object] = {
            k: v for k, v in asdict(self).items() if k not in ("n_estimators",)
        }
        if self.objective != "tweedie":
            params.pop("tweedie_variance_power")
        return params | {
            "bagging_freq": 1,
            "seed": seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": num_threads,
            "verbosity": -1,
        }


@dataclass
class LgbmPredictor:
    """LightGBM predictor for the walk-forward harness.

    Args:
        params: Hyperparameters for every fit.
        kickoff: Also use the ``ko_*`` kickoff-information columns (ablation only).
        seed: Seed for bagging and feature subsampling.
        num_threads: Fixed so results don't depend on the machine's core count.
    """

    params: LgbmParams = field(default_factory=LgbmParams)
    kickoff: bool = False
    seed: int = 20261025
    num_threads: int = 4
    columns: list[str] = field(default_factory=list, init=False)
    booster: lgb.Booster | None = field(default=None, init=False)

    @property
    def name(self) -> str:
        """``lgbm``, or ``lgbm_kickoff`` for the ablation."""
        return "lgbm_kickoff" if self.kickoff else "lgbm"

    def fit(self, train: pl.DataFrame) -> None:
        """Fit on ``train`` rows (features plus ``label_ppr``).

        Raises:
            ValueError: If there are no training rows.
        """
        if train.is_empty():
            raise ValueError(f"{self.name}: no training rows")
        self.columns = feature_columns(train.drop(LABEL), kickoff=self.kickoff)
        data = lgb.Dataset(
            to_matrix(train, self.columns),
            label=train[LABEL].cast(pl.Float64).to_numpy(),
            feature_name=self.columns,
            categorical_feature=categorical_indices(self.columns),
            free_raw_data=True,
        )
        self.booster = lgb.train(
            self.params.booster_params(self.seed, self.num_threads),
            data,
            num_boost_round=self.params.n_estimators,
        )

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Projected PPR points, clipped at 0.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self.booster is None:
            raise RuntimeError(f"{self.name}: predict called before fit")
        raw = np.asarray(self.booster.predict(to_matrix(features, self.columns)), dtype=np.float64)
        return pl.Series("prediction", np.clip(raw, 0.0, None))

    def importance(self) -> pl.DataFrame:
        """Total gain per feature from the latest fit, largest first."""
        if self.booster is None:
            raise RuntimeError(f"{self.name}: importance called before fit")
        gain = self.booster.feature_importance(importance_type="gain")
        return pl.DataFrame({"feature": self.columns, "gain": gain.astype(np.float64)}).sort(
            "gain", descending=True
        )
