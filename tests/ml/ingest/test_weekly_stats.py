import polars as pl
import pytest

from ml.ingest import common
from ml.ingest import weekly_stats as ws


def make_stats(season: int = 2021, weeks: list[int] | None = None) -> pl.DataFrame:
    weeks = weeks or [1, 2, 18, 19]
    n = len(weeks)
    reg = common.regular_season_weeks(season)
    return pl.DataFrame(
        {
            "player_id": [f"00-00{i:05d}" for i in range(n)],
            "player_display_name": [f"Player {i}" for i in range(n)],
            "position": ["WR"] * n,
            "season": [season] * n,
            "week": weeks,
            "season_type": ["REG" if w <= reg else "POST" for w in weeks],
            "game_id": [f"{season}_{w:02d}_AAA_BBB" for w in weeks],
            "team": ["AAA"] * n,
            "opponent_team": ["BBB"] * n,
            "fantasy_points": [10.0] * n,
            "fantasy_points_ppr": [12.5] * n,
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )


def test_validate_accepts_well_formed_frame() -> None:
    df = make_stats()
    assert ws.validate_weekly_stats(df, 2021).equals(df)


def test_postseason_is_tagged_post() -> None:
    df = make_stats().with_columns(pl.lit("WC").alias("season_type"))
    with pytest.raises(common.DataValidationError, match="unexpected season_type"):
        ws.validate_weekly_stats(df, 2021)


def test_same_player_in_regular_and_postseason_is_not_a_duplicate() -> None:
    df = make_stats(weeks=[18, 19]).with_columns(pl.lit("00-0000001").alias("player_id"))
    ws.validate_weekly_stats(df, 2021)


def test_duplicate_player_week_is_rejected() -> None:
    df = make_stats()
    with pytest.raises(common.DataValidationError, match="duplicate"):
        ws.validate_weekly_stats(pl.concat([df, df.head(1)]), 2021)


@pytest.mark.network
def test_fetch_real_season_passes_validation() -> None:
    df = ws.validate_weekly_stats(ws.fetch_weekly_stats(2023), 2023)
    assert df.height > 10_000
