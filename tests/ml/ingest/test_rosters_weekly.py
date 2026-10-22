import dataclasses
from pathlib import Path

import polars as pl
import pytest

from ml.ingest import common
from ml.ingest import rosters_weekly as rw


def make_rosters(season: int = 2021) -> pl.DataFrame:
    rows = [
        ("00-0000001", 1, "REG", "ACT"),
        ("00-0000001", 2, "REG", "RES"),
        (None, 1, "REG", "ACT"),
    ]
    rows += [("00-0000002", 19, "WC", "ACT")]
    n = len(rows)
    return pl.DataFrame(
        {
            "season": [season] * n,
            "week": [w for _, w, _, _ in rows],
            "game_type": [t for _, _, t, _ in rows],
            "team": ["AAA"] * n,
            "gsis_id": [g for g, _, _, _ in rows],
            "full_name": [f"Player {i}" for i in range(n)],
            "position": ["WR"] * n,
            "depth_chart_position": ["WR"] * n,
            "status": [s for _, _, _, s in rows],
            "years_exp": [3] * n,
            "pfr_id": ["PlayAb00"] * n,
            "sleeper_id": ["1234"] * n,
            "espn_id": ["5678"] * n,
            "yahoo_id": ["9012"] * n,
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32, "years_exp": pl.Int32},
    )


def validate(df: pl.DataFrame, season: int = 2021) -> pl.DataFrame:
    return common.validate(df, season, rw.SPEC)


def test_accepts_weekly_rows_with_some_null_ids() -> None:
    df = make_rosters()
    assert validate(df).equals(df)


def test_upstream_id_collision_is_kept_and_counted(tmp_path: Path) -> None:
    # Mirrors 2019: a second player listed under an existing gsis_id on another team.
    df = make_rosters()
    collision = df.head(1).with_columns(
        pl.lit("BBB").alias("team"), pl.lit("Other Player").alias("full_name")
    )
    spec = dataclasses.replace(rw.SPEC, fetch=lambda _: pl.concat([df, collision]))
    [record] = common.ingest_seasons(spec, [2021], tmp_path)
    assert (record.rows, record.null_id_rows, record.duplicate_key_rows) == (5, 1, 1)


def test_rejects_missing_platform_id_column() -> None:
    with pytest.raises(common.DataValidationError, match="missing columns"):
        validate(make_rosters().drop("sleeper_id"))


@pytest.mark.network
def test_fetch_real_season_passes_validation() -> None:
    df = validate(rw.fetch_rosters_weekly(2023), 2023)
    assert df.height > 40_000
