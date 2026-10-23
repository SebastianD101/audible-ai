import polars as pl
import pytest

from ml.features import usage
from ml.features.asof import Week
from tests.ml.features import synthetic as syn
from tests.ml.features.lookahead import assert_no_lookahead


def features(as_of: Week) -> pl.DataFrame:
    return usage.usage_features(syn.stats(), syn.snaps(), as_of)


def row(df: pl.DataFrame, player: str) -> dict[str, object]:
    return df.filter(pl.col("player_id") == player).row(0, named=True)


def test_in_season_rolling_skips_missed_games_and_byes() -> None:
    # wr1 2023 before week 5: played weeks 1 and 2 (missed week 3, bye week 4).
    wr1 = row(features(Week(2023, 5)), "wr1")
    assert wr1["games_played"] == 2
    assert wr1["ppr_season"] == pytest.approx(11.5)
    assert wr1["ppr_last3"] == pytest.approx(11.5)
    assert wr1["weeks_since_last_game"] == 3
    assert wr1["snap_pct_season"] == pytest.approx(0.8)


def test_last3_uses_only_the_three_most_recent_games() -> None:
    # wr1 2022 before week 6 played weeks 1, 2, 3, 5 (bye week 4): 11, 12, 13, 15 PPR.
    wr1 = row(features(Week(2022, 6)), "wr1")
    assert wr1["ppr_season"] == pytest.approx(12.75)
    assert wr1["ppr_last3"] == pytest.approx((12 + 13 + 15) / 3)
    assert wr1["ppr_trend"] == pytest.approx((12 + 13 + 15) / 3 - 12.75)


def test_rush_share_is_share_of_team_carries() -> None:
    assert row(features(Week(2023, 3)), "rb1")["rush_share_season"] == pytest.approx(1.0)
    assert row(features(Week(2023, 3)), "wr1")["rush_share_season"] == pytest.approx(0.0)


def test_prior_season_block() -> None:
    wr1 = row(features(Week(2023, 2)), "wr1")
    assert wr1["prev_games"] == 4
    assert wr1["prev_ppr"] == pytest.approx((11 + 12 + 13 + 15) / 4)


def test_player_with_only_prior_season_has_zero_games_played() -> None:
    old = row(features(Week(2023, 2)), "old")
    assert old["games_played"] == 0
    assert old["ppr_season"] is None
    assert old["prev_ppr"] == pytest.approx(4.0)


def test_usage_does_not_look_ahead() -> None:
    assert_no_lookahead(
        usage.usage_features, {"stats": syn.stats(), "snaps": syn.snaps()}, Week(2023, 3)
    )
