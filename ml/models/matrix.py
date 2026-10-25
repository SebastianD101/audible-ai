"""Turn a feature table into the numeric matrix a tree model trains on.

Identifiers and labels are never features. String categoricals get **fixed** integer codes
from a hard-coded vocabulary, so a code means the same thing in every weekly refit no matter
which categories happen to appear in that week's training rows. Nulls become NaN and are left
to the model (LightGBM routes missing values natively).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from ml.features.spine import LABEL_COLUMNS, SPINE_COLUMNS

KICKOFF_PREFIX = "ko_"

# Rare roster statuses (SUS, EXE, RSR, RET, TRC, TRD: under 0.1% of rows) share the last code.
OTHER = "other"
CATEGORICAL_VOCAB: Mapping[str, tuple[str, ...]] = {
    "position": ("QB", "RB", "WR", "TE"),
    "roster_status_last_week": ("ACT", "INA", "RES", "DEV", "CUT", OTHER),
}

# `last_report_status` is a one-to-one relabeling of `last_report_severity`.
REDUNDANT = frozenset({"last_report_status"})
NON_FEATURES = frozenset(SPINE_COLUMNS + LABEL_COLUMNS) - {"position"}


def feature_columns(df: pl.DataFrame, *, kickoff: bool = False) -> list[str]:
    """Model input columns of ``df``, in table order.

    Args:
        df: A feature table (labels may or may not be present).
        kickoff: Include the ``ko_*`` kickoff-information columns (ablation only).
    """
    return [
        c
        for c in df.columns
        if c not in NON_FEATURES
        and c not in REDUNDANT
        and (kickoff or not c.startswith(KICKOFF_PREFIX))
    ]


def categorical_indices(columns: Sequence[str]) -> list[int]:
    """Positions of the categorical columns within ``columns``."""
    return [i for i, c in enumerate(columns) if c in CATEGORICAL_VOCAB]


def encode(column: str, values: pl.Expr) -> pl.Expr:
    """Fixed integer code for a categorical; values outside the vocabulary map to ``other``."""
    vocab = CATEGORICAL_VOCAB[column]
    codes = {v: float(i) for i, v in enumerate(vocab)}
    default = codes.get(OTHER, -1.0)  # LightGBM treats negative categories as missing.
    return values.replace_strict(codes, default=default, return_dtype=pl.Float64)


def to_matrix(df: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """``df[columns]`` as a float64 array; categoricals encoded, nulls as NaN.

    Raises:
        KeyError: If a column is missing from ``df``.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns missing from frame: {missing}")
    exprs = [
        pl.when(pl.col(c).is_null()).then(None).otherwise(encode(c, pl.col(c))).alias(c)
        if c in CATEGORICAL_VOCAB
        else pl.col(c).cast(pl.Float64)
        for c in columns
    ]
    return df.select(exprs).to_numpy().astype(np.float64)
