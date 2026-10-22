import polars as pl
import pytest

from ml.ingest import common
from ml.ingest import snap_counts as sc


def make_snaps(season: int = 2021) -> pl.DataFrame:
    rows = [("SmitJo00", 1, "REG", 50.0, 0.8), ("DoeJa00", 1, "REG", 0.0, 0.0)]
    rows += [("SmitJo00", 19, "WC", 62.0, 1.0)]
    n = len(rows)
    return pl.DataFrame(
        {
            "game_id": [f"{season}_{w:02d}_AAA_BBB" for _, w, _, _, _ in rows],
            "season": [season] * n,
            "game_type": [t for _, _, t, _, _ in rows],
            "week": [w for _, w, _, _, _ in rows],
            "player": ["Player"] * n,
            "pfr_player_id": [p for p, _, _, _, _ in rows],
            "position": ["WR"] * n,
            "team": ["AAA"] * n,
            "opponent": ["BBB"] * n,
            "offense_snaps": [s for _, _, _, s, _ in rows],
            "offense_pct": [p for _, _, _, _, p in rows],
            "st_pct": [0.0, 1.01, 0.1],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )


def validate(df: pl.DataFrame, season: int = 2021) -> pl.DataFrame:
    return common.validate(df, season, sc.SPEC)


def test_accepts_snaps_including_rounded_special_teams_share() -> None:
    df = make_snaps()
    assert validate(df).equals(df)


@pytest.mark.parametrize("pct", [-0.1, 1.2])
def test_rejects_offense_pct_outside_unit_interval(pct: float) -> None:
    df = make_snaps().with_columns(pl.lit(pct).alias("offense_pct"))
    with pytest.raises(common.DataValidationError, match="offense_pct outside"):
        validate(df)


def test_rejects_same_player_twice_in_one_game() -> None:
    df = make_snaps()
    with pytest.raises(common.DataValidationError, match="duplicate"):
        validate(pl.concat([df, df.head(1)]))


@pytest.mark.network
def test_fetch_real_season_passes_validation() -> None:
    df = validate(sc.fetch_snap_counts(2023), 2023)
    assert df.height > 20_000
