"""Reusable leakage check: a feature's output must not depend on data at or after ``as_of``."""

from collections.abc import Callable

import polars as pl
from polars.testing import assert_frame_equal

from ml.features.asof import Week, before_expr

TIME_COLUMNS = frozenset({"season", "week"})


def corrupt_future(df: pl.DataFrame, as_of: Week, *, strings: bool) -> pl.DataFrame:
    """Scramble every row at or after ``as_of`` and append fake future copies of past rows.

    Numbers become large nonsense values. With ``strings=True``, strings become ``"FUTURE"``
    too; with ``strings=False`` they are kept, so IDs and team codes still join and a leak
    through a keyed lookup is caught. Season and week are never changed, so corrupted rows stay
    in the future. Frames without season/week columns (e.g. ID crosswalks) are returned as is.
    """
    if not set(df.columns) >= TIME_COLUMNS:
        return df

    def scramble(col: str, dtype: pl.DataType) -> pl.Expr:
        if col in TIME_COLUMNS:
            return pl.col(col)
        if dtype.is_numeric():
            return (pl.col(col).cast(pl.Float64).fill_null(0) * -7 + 999).cast(dtype).alias(col)
        if dtype == pl.String and strings:
            return pl.lit("FUTURE", dtype=pl.String).alias(col)
        return pl.col(col)

    garbage = [scramble(c, t) for c, t in df.schema.items()]
    future = ~before_expr(as_of)
    scrambled = df.with_columns(
        pl.when(future).then(g).otherwise(pl.col(c)).alias(c)
        for g, c in zip(garbage, df.columns, strict=True)
    )
    fake = df.filter(~future).tail(5).with_columns(garbage)
    fake_rows = [
        fake.with_columns(
            pl.lit(as_of.season, dtype=df.schema["season"]).alias("season"),
            pl.lit(as_of.week + offset, dtype=df.schema["week"]).alias("week"),
        )
        for offset in (0, 1)
    ]
    return pl.concat([scrambled, *fake_rows])


def assert_no_lookahead(
    fn: Callable[..., pl.DataFrame],
    inputs: dict[str, pl.DataFrame],
    as_of: Week,
    *,
    known_in_advance: frozenset[str] = frozenset(),
) -> pl.DataFrame:
    """Assert ``fn(**inputs, as_of=as_of)`` is unchanged when future rows are corrupted.

    Runs two corruptions (numbers only, then numbers and strings). ``known_in_advance`` names
    inputs exempt from corruption because they are published before the season, i.e. the
    schedule's fixtures (never its results). Returns the clean output so tests can also check
    values. Fails on an empty output, which would pass vacuously.
    """
    clean = fn(**inputs, as_of=as_of)
    assert clean.height, "feature output is empty; the lookahead check would be vacuous"
    for strings in (False, True):
        corrupted = {
            k: v if k in known_in_advance else corrupt_future(v, as_of, strings=strings)
            for k, v in inputs.items()
        }
        assert_frame_equal(clean, fn(**corrupted, as_of=as_of), check_row_order=False)
    return clean
