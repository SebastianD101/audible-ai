import dataclasses
from pathlib import Path

import polars as pl
import pytest

from ml.ingest import common
from ml.ingest import injuries as inj


def make_injuries(season: int = 2021, *, legacy: bool = False) -> pl.DataFrame:
    """Build an injury report; ``legacy`` mimics 2018-2020 (float keys, ``date_modified``)."""
    rows = [
        ("00-0000001", 1, "REG", "Questionable", "Limited Participation in Practice"),
        ("00-0000002", 1, "REG", "Out", "Did Not Participate In Practice"),
        ("00-0000001", 19, "WC", None, "\n    "),
    ]
    n = len(rows)
    df = pl.DataFrame(
        {
            "season": [season] * n,
            "week": [w for _, w, _, _, _ in rows],
            "game_type": [t for _, _, t, _, _ in rows],
            "team": ["AAA"] * n,
            "gsis_id": [g for g, _, _, _, _ in rows],
            "full_name": ["Player"] * n,
            "position": ["RB"] * n,
            "report_primary_injury": ["Ankle"] * n,
            "report_status": [r for _, _, _, r, _ in rows],
            "practice_status": [p for _, _, _, _, p in rows],
        },
        schema_overrides={"report_status": pl.String},
    )
    if legacy:
        return df.with_columns(
            pl.col("season", "week").cast(pl.Float64), pl.lit("2018-09-07").alias("date_modified")
        )
    return df.with_columns(
        pl.col("season", "week").cast(pl.Int32),
        pl.when(pl.col("game_type") == "REG")
        .then(pl.lit("REG"))
        .otherwise(pl.lit("POST"))
        .alias("season_type"),
    )


def ingest(tmp_path: Path, frame: pl.DataFrame, season: int = 2021) -> common.SeasonRecord:
    spec = dataclasses.replace(inj.SPEC, fetch=lambda _: frame)
    [record] = common.ingest_seasons(spec, [season], tmp_path)
    return record


@pytest.mark.parametrize(("season", "legacy"), [(2025, False), (2019, True)])
def test_both_schema_eras_ingest_to_int32_keys(tmp_path: Path, season: int, legacy: bool) -> None:
    ingest(tmp_path, make_injuries(season, legacy=legacy), season)
    stored = pl.read_parquet(common.season_path(tmp_path, season))
    assert stored.schema["season"] == stored.schema["week"] == pl.Int32
    assert ("date_modified" in stored.columns) is legacy
    assert ("season_type" in stored.columns) is not legacy


def test_mid_week_status_change_is_kept_and_counted(tmp_path: Path) -> None:
    df = make_injuries()
    updated = df.head(1).with_columns(pl.lit("Out").alias("report_status"))
    record = ingest(tmp_path, pl.concat([df, updated]))
    assert (record.rows, record.duplicate_key_rows) == (4, 1)


def test_unknown_report_status_is_rejected() -> None:
    df = make_injuries().with_columns(pl.lit("Probable").alias("report_status"))
    with pytest.raises(common.DataValidationError, match="unknown report_status"):
        common.validate(df, 2021, inj.SPEC)


def test_note_and_null_statuses_are_allowed() -> None:
    df = make_injuries().with_columns(
        pl.when(pl.col("week") == 1).then(pl.lit("Note")).alias("report_status")
    )
    common.validate(df, 2021, inj.SPEC)


@pytest.mark.network
@pytest.mark.parametrize("season", [2019, 2025])
def test_fetch_real_season_passes_validation(season: int) -> None:
    df = common.validate(inj.normalize_injuries(inj.fetch_injuries(season)), season, inj.SPEC)
    assert df.height > 4_000
