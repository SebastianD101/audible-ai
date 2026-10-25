import numpy as np
import polars as pl
import pytest

from ml.features.spine import LABEL_COLUMNS
from ml.models import matrix
from tests.ml.backtest.frames import feature_table


def with_extras(df: pl.DataFrame) -> pl.DataFrame:
    n = df.height
    return df.with_columns(
        pl.Series("roster_status_last_week", (["ACT", "RES", "SUS", None] * n)[:n]),
        pl.Series("last_report_status", ["Out"] * n),
        pl.Series("last_report_severity", [3] * n),
        pl.Series("missed_last_game", ([True, False] * n)[:n]),
        pl.Series("ko_total_line", [44.5] * n),
    )


def test_feature_columns_exclude_ids_labels_redundant_and_kickoff() -> None:
    cols = matrix.feature_columns(with_extras(feature_table()))
    assert "position" in cols
    for dropped in ["player_id", "name", "team", "season", "week", "last_report_status"]:
        assert dropped not in cols
    assert not set(LABEL_COLUMNS) & set(cols)
    assert "ko_total_line" not in cols
    assert "ko_total_line" in matrix.feature_columns(with_extras(feature_table()), kickoff=True)


def test_codes_are_fixed_regardless_of_which_categories_appear() -> None:
    df = with_extras(feature_table())
    cols = ["position", "roster_status_last_week"]
    full = matrix.to_matrix(df, cols)
    only_te = matrix.to_matrix(df.filter(pl.col("position") == "TE"), cols)
    assert set(only_te[:, 0]) == {3.0}
    assert set(full[:, 0]) == {0.0, 1.0, 2.0, 3.0}
    statuses = full[:4, 1]
    assert statuses[0] == 0.0  # ACT
    assert statuses[1] == 2.0  # RES
    assert statuses[2] == 5.0  # SUS -> other
    assert np.isnan(statuses[3])  # null stays missing


def test_booleans_and_ints_become_floats_and_input_is_untouched() -> None:
    df = with_extras(feature_table())
    before = df.clone()
    x = matrix.to_matrix(df, ["missed_last_game", "last_report_severity", "ppr_season"])
    assert x.dtype == np.float64
    assert set(x[:, 0]) == {0.0, 1.0}
    assert df.equals(before)


def test_categorical_indices() -> None:
    assert matrix.categorical_indices(["ppr_season", "position", "rest"]) == [1]


def test_missing_column_raises() -> None:
    with pytest.raises(KeyError, match="nope"):
        matrix.to_matrix(feature_table(), ["nope"])
