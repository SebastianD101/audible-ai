"""Which rows a metric is computed over.

About 44% of spine rows score 0 (backups, inactive players), which makes RMSE on the full spine
look better than the model's usefulness warrants. Every metric is therefore reported on:

- ``all``: every spine row.
- ``relevant``: per (week, position), the top N players by a **pre-week** score (season PPG,
  falling back to prior-season PPG). N is QB 24, RB 48, WR 60, TE 24, roughly the players a
  12-team league starts or benches. Membership uses no labels and no model output, so every
  model is evaluated on the same rows.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

RELEVANT_PER_POSITION: Mapping[str, int] = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
SCOPES = ("all", "relevant")
KEYS = ["player_id", "season", "week"]


def relevance(
    features: pl.DataFrame, limits: Mapping[str, int] = RELEVANT_PER_POSITION
) -> pl.DataFrame:
    """Return ``KEYS`` plus a boolean ``relevant`` column, from pre-week columns only."""
    score = pl.coalesce(pl.col("ppr_season"), pl.col("prev_ppr"), pl.lit(0.0))
    limit = pl.col("position").replace_strict(dict(limits), default=0, return_dtype=pl.Int64)
    ranked = features.select(*KEYS, "position", score.alias("score")).sort(
        "score", "player_id", descending=[True, False]
    )
    rank = pl.int_range(1, pl.len() + 1).over("season", "week", "position")
    return ranked.select(*KEYS, (rank <= limit).alias("relevant"))


def attach_scopes(predictions: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """Add the ``relevant`` flag to a predictions frame."""
    return predictions.join(relevance(features), on=KEYS, how="left").with_columns(
        pl.col("relevant").fill_null(False)
    )


def in_scope(predictions: pl.DataFrame, scope: str) -> pl.DataFrame:
    """Rows of ``predictions`` belonging to ``scope``.

    Raises:
        ValueError: For an unknown scope name.
    """
    if scope == "all":
        return predictions
    if scope == "relevant":
        return predictions.filter(pl.col("relevant"))
    raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")
