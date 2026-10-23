import polars as pl
import pytest

from ml.backtest import scopes
from tests.ml.backtest.frames import feature_table

LIMITS = {"QB": 1, "RB": 2, "WR": 2, "TE": 0}


def test_relevant_keeps_top_n_per_position_and_week_by_pre_week_score() -> None:
    out = scopes.relevance(feature_table(), LIMITS).join(
        feature_table().select(*scopes.KEYS, "position"), on=scopes.KEYS
    )
    picked = out.filter(pl.col("relevant"), pl.col("season") == 2022, pl.col("week") == 3)
    assert sorted(picked["player_id"]) == ["q3", "r2", "r3", "w2", "w3"]
    per_week = out.filter(pl.col("relevant")).group_by("season", "week").len()["len"]
    assert set(per_week) == {5}


def test_prior_season_score_used_when_no_games_this_season() -> None:
    df = feature_table().with_columns(
        pl.when(pl.col("player_id") == "q1")
        .then(None)
        .otherwise(pl.col("ppr_season"))
        .alias("ppr_season"),
        pl.when(pl.col("player_id") == "q1")
        .then(99.0)
        .otherwise(pl.col("prev_ppr"))
        .alias("prev_ppr"),
    )
    out = scopes.relevance(df, LIMITS).filter(pl.col("relevant"))
    assert "q1" in set(out["player_id"])
    assert "q3" not in set(out["player_id"])


def test_relevance_ignores_labels() -> None:
    df = feature_table()
    scrambled = df.with_columns((pl.col("label_ppr") * -13 + 7).alias("label_ppr"))
    assert scopes.relevance(df).equals(scopes.relevance(scrambled))


def test_attach_and_filter_scopes() -> None:
    df = feature_table()
    preds = df.select(*scopes.KEYS, "label_ppr").with_columns(pl.lit(1.0).alias("prediction"))
    tagged = scopes.attach_scopes(
        preds, df.select(*scopes.KEYS, "position", "ppr_season", "prev_ppr")
    )
    assert scopes.in_scope(tagged, "all").height == df.height
    # Default limits exceed this small league, so every row is relevant.
    assert scopes.in_scope(tagged, "relevant").height == df.height
    with pytest.raises(ValueError, match="unknown scope"):
        scopes.in_scope(tagged, "starters")
