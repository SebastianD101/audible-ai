import polars as pl
import pytest

from ml.features.asof import Week, available_before
from tests.ml.features.lookahead import assert_no_lookahead, corrupt_future


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2022, 2023, 2023, 2023, 2024],
            "week": [18, 9, 10, 11, 1],
            "player_id": ["a"] * 5,
            "points": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_week_orders_by_season_then_week() -> None:
    assert Week(2022, 18) < Week(2023, 1) < Week(2023, 2)
    assert str(Week(2023, 9)) == "2023-W09"


def test_week_rejects_week_zero() -> None:
    with pytest.raises(ValueError, match="week must be"):
        Week(2023, 0)


def test_available_before_is_strict_and_spans_seasons() -> None:
    out = available_before(frame(), Week(2023, 10))
    assert out.select("season", "week").rows() == [(2022, 18), (2023, 9)]


def test_available_before_works_lazily() -> None:
    lazy = available_before(frame().lazy(), Week(2023, 10))
    assert lazy.collect().height == 2


@pytest.mark.parametrize("strings", [False, True])
def test_corrupt_future_leaves_past_rows_alone(strings: bool) -> None:
    as_of = Week(2023, 10)
    corrupted = corrupt_future(frame(), as_of, strings=strings)
    assert available_before(corrupted, as_of).equals(available_before(frame(), as_of))
    future = corrupted.filter(~pl.col("points").is_in([1.0, 2.0]))
    assert future.height == corrupted.height - 2
    assert set(future["player_id"]) == ({"FUTURE"} if strings else {"a"})


def test_assert_no_lookahead_catches_a_leak_through_a_keyed_join() -> None:
    def leaky_join(stats: pl.DataFrame, as_of: Week) -> pl.DataFrame:
        past = available_before(stats, as_of).group_by("player_id").agg(pl.col("points").sum())
        this_week = stats.filter(pl.col("season") == as_of.season, pl.col("week") == as_of.week)
        return past.join(this_week.select("player_id", "week"), on="player_id", how="left")

    with pytest.raises(AssertionError):
        assert_no_lookahead(leaky_join, {"stats": frame()}, Week(2023, 10))


def test_assert_no_lookahead_catches_a_leaky_feature() -> None:
    def leaky(stats: pl.DataFrame, as_of: Week) -> pl.DataFrame:
        return stats.group_by("player_id").agg(pl.col("points").mean())

    with pytest.raises(AssertionError):
        assert_no_lookahead(leaky, {"stats": frame()}, Week(2023, 10))


def test_assert_no_lookahead_passes_a_safe_feature() -> None:
    def safe(stats: pl.DataFrame, as_of: Week) -> pl.DataFrame:
        return available_before(stats, as_of).group_by("player_id").agg(pl.col("points").mean())

    out = assert_no_lookahead(safe, {"stats": frame()}, Week(2023, 10))
    assert out["points"].item() == pytest.approx(1.5)
