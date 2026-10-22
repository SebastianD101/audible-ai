import json
from pathlib import Path

import polars as pl
import pytest

from ml.ingest import weekly_stats as ws


def make_stats(season: int = 2021, weeks: list[int] | None = None) -> pl.DataFrame:
    weeks = weeks or [1, 2, 18, 19]
    n = len(weeks)
    reg = ws.regular_season_weeks(season)
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


@pytest.mark.parametrize(("season", "expected"), [(2018, 17), (2020, 17), (2021, 18), (2025, 18)])
def test_regular_season_weeks(season: int, expected: int) -> None:
    assert ws.regular_season_weeks(season) == expected


def test_validate_accepts_well_formed_frame() -> None:
    df = make_stats()
    assert ws.validate_weekly_stats(df, 2021).equals(df)


def test_validate_allows_null_player_id_rows() -> None:
    df = make_stats().with_columns(pl.lit(None, dtype=pl.String).alias("player_id"))
    ws.validate_weekly_stats(df, 2021)


def test_validate_rejects_missing_column() -> None:
    with pytest.raises(ws.DataValidationError, match="missing columns"):
        ws.validate_weekly_stats(make_stats().drop("fantasy_points_ppr"), 2021)


def test_validate_rejects_wrong_dtype() -> None:
    df = make_stats().with_columns(pl.col("week").cast(pl.String))
    with pytest.raises(ws.DataValidationError, match="unexpected column types"):
        ws.validate_weekly_stats(df, 2021)


def test_validate_rejects_empty_frame() -> None:
    with pytest.raises(ws.DataValidationError, match="no rows"):
        ws.validate_weekly_stats(make_stats().clear(), 2021)


def test_validate_rejects_wrong_season() -> None:
    with pytest.raises(ws.DataValidationError, match="found seasons"):
        ws.validate_weekly_stats(make_stats(season=2021), 2022)


def test_validate_rejects_unknown_season_type() -> None:
    df = make_stats().with_columns(pl.lit("PRE").alias("season_type"))
    with pytest.raises(ws.DataValidationError, match="season_type"):
        ws.validate_weekly_stats(df, 2021)


def test_validate_rejects_week_18_in_17_week_season() -> None:
    df = make_stats(season=2020, weeks=[1, 17]).with_columns(
        pl.Series("week", [1, 18], dtype=pl.Int32), pl.lit("REG").alias("season_type")
    )
    with pytest.raises(ws.DataValidationError, match="weeks out of range"):
        ws.validate_weekly_stats(df, 2020)


def test_validate_rejects_postseason_week_beyond_super_bowl() -> None:
    with pytest.raises(ws.DataValidationError, match="weeks out of range"):
        ws.validate_weekly_stats(make_stats(weeks=[1, 23]), 2021)


def test_validate_rejects_duplicate_keys() -> None:
    df = make_stats()
    with pytest.raises(ws.DataValidationError, match="duplicate"):
        ws.validate_weekly_stats(pl.concat([df, df.head(1)]), 2021)


def test_write_season_round_trips(tmp_path: Path) -> None:
    df = make_stats()
    path = ws.write_season(df, tmp_path, 2021)
    assert path == tmp_path / "season=2021.parquet"
    assert pl.read_parquet(path).equals(df)
    assert not list(tmp_path.glob("*.tmp"))


def test_ingest_skips_existing_unless_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake_fetch(season: int) -> pl.DataFrame:
        calls.append(season)
        return make_stats(season=season)

    monkeypatch.setattr(ws, "fetch_weekly_stats", fake_fetch)

    ws.ingest_seasons([2020, 2021], tmp_path)
    assert calls == [2020, 2021]

    ws.ingest_seasons([2020, 2021], tmp_path)
    assert calls == [2020, 2021]

    ws.ingest_seasons([2021], tmp_path, force=True)
    assert calls == [2020, 2021, 2021]


def test_ingest_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws, "fetch_weekly_stats", lambda season: make_stats(season=season))

    records = ws.ingest_seasons([2021, 2020], tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [s["season"] for s in manifest["seasons"]] == [2020, 2021]
    assert manifest["nflreadpy_version"]
    for entry, record in zip(
        manifest["seasons"], sorted(records, key=lambda r: r.season), strict=True
    ):
        assert entry["rows"] == 4
        assert entry["sha256"] == ws.sha256_file(tmp_path / entry["file"]) == record.sha256


def test_partial_run_keeps_other_seasons_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ws, "fetch_weekly_stats", lambda season: make_stats(season=season))
    ws.ingest_seasons([2020, 2021], tmp_path)

    ws.ingest_seasons([2021], tmp_path, force=True)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [s["season"] for s in manifest["seasons"]] == [2020, 2021]


def test_ingest_does_not_write_invalid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ws, "fetch_weekly_stats", lambda season: make_stats(season=season + 1))
    with pytest.raises(ws.DataValidationError):
        ws.ingest_seasons([2021], tmp_path)
    assert not ws.season_path(tmp_path, 2021).exists()


@pytest.mark.parametrize(
    "argv", [["--start-season", "2026", "--end-season", "2025"], ["--start-season", "1990"]]
)
def test_parse_args_rejects_bad_ranges(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        ws.parse_args(argv)


def test_parse_args_defaults_exclude_holdout_season() -> None:
    args = ws.parse_args([])
    assert (args.start_season, args.end_season) == (2018, 2025)


@pytest.mark.network
def test_fetch_real_season_passes_validation() -> None:
    df = ws.validate_weekly_stats(ws.fetch_weekly_stats(2023), 2023)
    assert df.height > 10_000
