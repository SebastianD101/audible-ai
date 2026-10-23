"""Point metrics and week-block bootstrap confidence intervals.

Metrics (CLAUDE.md): RMSE and MAE on PPR points, and Spearman rank correlation computed **within
each (week, position)** and averaged. Ranking players of one position against each other in a
given week is the decision that matters; a pooled correlation would mostly reward knowing that
QBs outscore TEs.

Uncertainty: rows in the same week share game environments, so resampling rows would overstate
precision. The bootstrap resamples whole *weeks* with replacement. Each week is first reduced
to sufficient statistics (error sums, row counts, Spearman sums over its position groups), so a
draw is a weighted sum over weeks, and B draws are one matrix product. Paired comparisons apply
the same draws to both models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

METRICS = ("rmse", "mae", "spearman")
DEFAULT_BOOTSTRAP = 1000
DEFAULT_SEED = 20261023
MIN_GROUP_SIZE = 3  # Spearman needs a few players per (week, position) to mean anything.


@dataclass(frozen=True)
class Estimate:
    """A metric's point estimate, 95% percentile interval, and sample size."""

    metric: str
    point: float
    lo: float
    hi: float
    n_weeks: int
    n_rows: int


def week_stats(preds: pl.DataFrame) -> pl.DataFrame:
    """Reduce predictions to one row of sufficient statistics per (season, week)."""
    errors = preds.with_columns((pl.col("prediction") - pl.col("label_ppr")).alias("err"))
    by_week = errors.group_by("season", "week").agg(
        (pl.col("err") ** 2).sum().alias("sq_err"),
        pl.col("err").abs().sum().alias("abs_err"),
        pl.len().alias("n"),
    )
    groups = (
        preds.group_by("season", "week", "position")
        .agg(
            pl.len().alias("size"),
            pl.corr("prediction", "label_ppr", method="spearman").alias("rho"),
        )
        .filter(pl.col("size") >= MIN_GROUP_SIZE, pl.col("rho").is_finite())
        .group_by("season", "week")
        .agg(pl.col("rho").sum().alias("rho_sum"), pl.len().alias("rho_n"))
    )
    return (
        by_week.join(groups, on=["season", "week"], how="left")
        .with_columns(pl.col("rho_sum").fill_null(0.0), pl.col("rho_n").fill_null(0))
        .sort("season", "week")
    )


def metrics_from_weights(stats: pl.DataFrame, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Compute every metric for each row of ``weights`` (draws x weeks)."""

    def col(name: str) -> np.ndarray:
        return stats[name].to_numpy().astype(np.float64)

    n = weights @ col("n")
    rho_n = weights @ col("rho_n")
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "rmse": np.sqrt((weights @ col("sq_err")) / n),
            "mae": (weights @ col("abs_err")) / n,
            "spearman": np.where(rho_n > 0, (weights @ col("rho_sum")) / rho_n, np.nan),
        }


def week_draws(n_weeks: int, n_boot: int, seed: int) -> np.ndarray:
    """Return a (n_boot x n_weeks) matrix of how often each week appears in each draw."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_weeks, size=(n_boot, n_weeks))
    return np.stack([np.bincount(row, minlength=n_weeks) for row in picks]).astype(np.float64)


def summarize(
    point: dict[str, np.ndarray], draws: dict[str, np.ndarray], n_weeks: int, n_rows: int
) -> dict[str, Estimate]:
    """Turn point values and bootstrap draws into 95% percentile ``Estimate``s."""
    out = {}
    for metric in METRICS:
        values = draws[metric][np.isfinite(draws[metric])]
        lo, hi = np.quantile(values, [0.025, 0.975]) if values.size else (np.nan, np.nan)
        out[metric] = Estimate(
            metric, float(point[metric][0]), float(lo), float(hi), n_weeks, n_rows
        )
    return out


def bootstrap(
    preds: pl.DataFrame, n_boot: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED
) -> dict[str, Estimate]:
    """Point estimates and week-block bootstrap intervals for one model's predictions."""
    stats = week_stats(preds)
    n_weeks = stats.height
    point = metrics_from_weights(stats, np.ones((1, n_weeks)))
    draws = metrics_from_weights(stats, week_draws(n_weeks, n_boot, seed))
    return summarize(point, draws, n_weeks, preds.height)


def paired_bootstrap(
    preds_a: pl.DataFrame,
    preds_b: pl.DataFrame,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Estimate]:
    """Metric of A minus metric of B, resampling the same weeks for both.

    Raises:
        ValueError: If the two frames don't cover the same player-weeks.
    """
    keys = ["player_id", "season", "week"]
    if not preds_a.select(keys).sort(keys).equals(preds_b.select(keys).sort(keys)):
        raise ValueError("paired comparison needs both models on the same player-weeks")
    a, b = week_stats(preds_a), week_stats(preds_b)
    weights_one = np.ones((1, a.height))
    weights = week_draws(a.height, n_boot, seed)
    point_a, point_b = metrics_from_weights(a, weights_one), metrics_from_weights(b, weights_one)
    draws_a, draws_b = metrics_from_weights(a, weights), metrics_from_weights(b, weights)
    point = {m: point_a[m] - point_b[m] for m in METRICS}
    draws = {m: draws_a[m] - draws_b[m] for m in METRICS}
    return summarize(point, draws, a.height, preds_a.height)
