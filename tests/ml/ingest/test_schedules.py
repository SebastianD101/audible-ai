import polars as pl
import pytest

from ml.ingest import common
from ml.ingest import schedules as sch


def make_schedule(season: int = 2021) -> pl.DataFrame:
    games = [(1, "REG", "AAA", "BBB"), (1, "REG", "CCC", "DDD"), (19, "WC", "AAA", "CCC")]
    n = len(games)
    return pl.DataFrame(
        {
            "game_id": [f"{season}_{w:02d}_{a}_{h}" for w, _, a, h in games],
            "season": [season] * n,
            "game_type": [t for _, t, _, _ in games],
            "week": [w for w, _, _, _ in games],
            "gameday": [f"{season}-09-12"] * n,
            "gametime": ["13:00"] * n,
            "away_team": [a for _, _, a, _ in games],
            "home_team": [h for _, _, _, h in games],
            "away_score": [20, 17, 24],
            "home_score": [27, 17, 10],
            "away_rest": [7] * n,
            "home_rest": [7] * n,
            "spread_line": [3.0, -1.5, 6.5],
            "total_line": [44.5, 41.0, 47.0],
            "away_moneyline": [140, -120, -280],
            "home_moneyline": [-160, 100, 230],
            "div_game": [1, 0, 0],
            "roof": ["outdoors", "dome", "outdoors"],
        },
        schema_overrides={
            col: pl.Int32
            for col in [
                "season",
                "week",
                "away_score",
                "home_score",
                "away_rest",
                "home_rest",
                "away_moneyline",
                "home_moneyline",
                "div_game",
            ]
        },
    )


def validate(df: pl.DataFrame, season: int = 2021) -> pl.DataFrame:
    return common.validate(df, season, sch.SPEC)


def test_accepts_regular_and_postseason_games() -> None:
    df = make_schedule()
    assert validate(df).equals(df)


def test_rejects_duplicate_game_id() -> None:
    df = make_schedule()
    with pytest.raises(common.DataValidationError, match="duplicate"):
        validate(pl.concat([df, df.head(1)]))


def test_rejects_played_game_without_lines() -> None:
    df = make_schedule().with_columns(
        pl.when(pl.col("week") == 19)
        .then(None)
        .otherwise(pl.col("spread_line"))
        .alias("spread_line")
    )
    with pytest.raises(common.DataValidationError, match="missing lines"):
        validate(df)


def test_allows_unplayed_game_without_lines() -> None:
    unplayed = pl.col("week") == 19
    df = make_schedule().with_columns(
        *(
            pl.when(unplayed).then(None).otherwise(pl.col(c)).alias(c)
            for c in ["home_score", "away_score", "spread_line", "total_line"]
        )
    )
    validate(df)


def test_rejects_team_with_two_games_in_one_week() -> None:
    df = make_schedule().with_columns(
        pl.when(pl.col("game_id") == "2021_01_CCC_DDD")
        .then(pl.lit("AAA"))
        .otherwise(pl.col("away_team"))
        .alias("away_team")
    )
    with pytest.raises(common.DataValidationError, match="two games in one week"):
        validate(df)


def test_rejects_postseason_week_in_regular_season_range() -> None:
    df = make_schedule().with_columns(
        pl.when(pl.col("game_type") == "WC")
        .then(pl.lit(18, dtype=pl.Int32))
        .otherwise(pl.col("week"))
        .alias("week")
    )
    with pytest.raises(common.DataValidationError, match="weeks out of range"):
        validate(df)


@pytest.mark.network
def test_fetch_real_season_passes_validation_and_spread_favors_home() -> None:
    df = validate(sch.fetch_schedules(2023), 2023)
    assert df.height == 285
    # Positive spread_line means the home team is favored, so it tracks the home margin.
    corr = df.select(pl.corr("spread_line", "result")).item()
    assert corr > 0.3
