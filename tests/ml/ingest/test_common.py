import argparse
import dataclasses
import json
from pathlib import Path

import polars as pl
import pytest

from ml.ingest import common


def make_frame(season: int = 2021, weeks: list[int] | None = None) -> pl.DataFrame:
    weeks = weeks or [1, 2, 18, 19]
    n = len(weeks)
    reg = common.regular_season_weeks(season)
    return pl.DataFrame(
        {
            "player_id": [f"00-00{i:05d}" for i in range(n)],
            "season": [season] * n,
            "week": weeks,
            "game_type": ["REG" if w <= reg else "WC" for w in weeks],
            "points": [10.0] * n,
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )


def fail_fetch(season: int) -> pl.DataFrame:
    raise AssertionError(f"unexpected fetch of {season}")


SPEC = common.DatasetSpec(
    name="fake",
    source="test fixture",
    fetch=fail_fetch,
    default_out_dir=Path("unused"),
    required_columns={
        "player_id": pl.String,
        "season": pl.Int32,
        "week": pl.Int32,
        "game_type": pl.String,
        "points": pl.Float64,
    },
    key_columns=["player_id", "week"],
    id_column="player_id",
    season_type_column="game_type",
    season_types=frozenset({"REG", "WC", "DIV", "CON", "SB"}),
)


def with_fetch(spec: common.DatasetSpec, calls: list[int] | None = None) -> common.DatasetSpec:
    def fetch(season: int) -> pl.DataFrame:
        if calls is not None:
            calls.append(season)
        return make_frame(season=season)

    return dataclasses.replace(spec, fetch=fetch)


@pytest.mark.parametrize(("season", "expected"), [(2018, 17), (2020, 17), (2021, 18), (2025, 18)])
def test_regular_season_weeks(season: int, expected: int) -> None:
    assert common.regular_season_weeks(season) == expected


def test_validate_accepts_well_formed_frame() -> None:
    df = make_frame()
    assert common.validate(df, 2021, SPEC).equals(df)


def test_validate_keeps_extra_columns() -> None:
    df = make_frame().with_columns(pl.lit("x").alias("new_in_2025"))
    assert "new_in_2025" in common.validate(df, 2021, SPEC).columns


def test_validate_allows_null_ids() -> None:
    df = make_frame().with_columns(pl.lit(None, dtype=pl.String).alias("player_id"))
    common.validate(df, 2021, SPEC)


def test_validate_rejects_missing_column() -> None:
    with pytest.raises(common.DataValidationError, match="missing columns"):
        common.validate(make_frame().drop("points"), 2021, SPEC)


def test_validate_rejects_wrong_dtype() -> None:
    df = make_frame().with_columns(pl.col("week").cast(pl.String))
    with pytest.raises(common.DataValidationError, match="unexpected column types"):
        common.validate(df, 2021, SPEC)


def test_validate_rejects_empty_frame() -> None:
    with pytest.raises(common.DataValidationError, match="no rows"):
        common.validate(make_frame().clear(), 2021, SPEC)


def test_validate_rejects_wrong_season() -> None:
    with pytest.raises(common.DataValidationError, match="found seasons"):
        common.validate(make_frame(season=2021), 2022, SPEC)


def test_validate_rejects_unknown_season_type() -> None:
    df = make_frame().with_columns(pl.lit("PRE").alias("game_type"))
    with pytest.raises(common.DataValidationError, match="unexpected game_type"):
        common.validate(df, 2021, SPEC)


def test_validate_rejects_week_18_in_17_week_season() -> None:
    df = make_frame(season=2020, weeks=[1, 17]).with_columns(
        pl.Series("week", [1, 18], dtype=pl.Int32), pl.lit("REG").alias("game_type")
    )
    with pytest.raises(common.DataValidationError, match="weeks out of range"):
        common.validate(df, 2020, SPEC)


def test_validate_rejects_postseason_week_beyond_super_bowl() -> None:
    with pytest.raises(common.DataValidationError, match="weeks out of range"):
        common.validate(make_frame(weeks=[1, 23]), 2021, SPEC)


def test_validate_rejects_duplicate_keys() -> None:
    df = make_frame()
    with pytest.raises(common.DataValidationError, match="duplicate"):
        common.validate(pl.concat([df, df.head(1)]), 2021, SPEC)


def test_validate_warn_mode_keeps_duplicate_keys() -> None:
    df = make_frame()
    doubled = pl.concat([df, df.head(1)])
    spec = dataclasses.replace(SPEC, duplicate_keys="warn")
    assert common.validate(doubled, 2021, spec).height == df.height + 1
    assert common.count_duplicate_keys(doubled, spec) == 1


def test_validate_runs_extra_checks() -> None:
    def no_negative_points(df: pl.DataFrame, season: int) -> None:
        if (df["points"] < 0).any():
            raise common.DataValidationError(f"season {season}: negative points")

    spec = dataclasses.replace(SPEC, checks=(no_negative_points,))
    common.validate(make_frame(), 2021, spec)
    with pytest.raises(common.DataValidationError, match="negative points"):
        common.validate(make_frame().with_columns(pl.lit(-1.0).alias("points")), 2021, spec)


def test_cast_integral_floats_casts_whole_numbers_only() -> None:
    df = make_frame().with_columns(pl.col("season", "week").cast(pl.Float64))
    out = common.cast_integral_floats(df, ["season", "week", "points", "not_a_column"])
    assert out.equals(make_frame().with_columns(pl.col("points").cast(pl.Int32)))


def test_cast_integral_floats_rejects_fractional_values() -> None:
    df = make_frame().with_columns(pl.lit(1.5).alias("points"))
    with pytest.raises(common.DataValidationError, match="non-integral"):
        common.cast_integral_floats(df, ["points"])


def test_ingest_normalizes_before_validating(tmp_path: Path) -> None:
    spec = dataclasses.replace(
        SPEC,
        fetch=lambda season: make_frame(season).with_columns(pl.col("week").cast(pl.Float64)),
        normalize=lambda df: common.cast_integral_floats(df, ["week"]),
    )
    common.ingest_seasons(spec, [2021], tmp_path)
    assert pl.read_parquet(common.season_path(tmp_path, 2021)).schema["week"] == pl.Int32


def test_write_season_round_trips(tmp_path: Path) -> None:
    df = make_frame()
    path = common.write_season(df, tmp_path, 2021)
    assert path == tmp_path / "season=2021.parquet"
    assert pl.read_parquet(path).equals(df)
    assert not list(tmp_path.glob("*.tmp"))


def test_ingest_skips_existing_unless_forced(tmp_path: Path) -> None:
    calls: list[int] = []
    spec = with_fetch(SPEC, calls)

    common.ingest_seasons(spec, [2020, 2021], tmp_path)
    assert calls == [2020, 2021]

    common.ingest_seasons(spec, [2020, 2021], tmp_path)
    assert calls == [2020, 2021]

    common.ingest_seasons(spec, [2021], tmp_path, force=True)
    assert calls == [2020, 2021, 2021]


def test_ingest_writes_manifest(tmp_path: Path) -> None:
    records = common.ingest_seasons(with_fetch(SPEC), [2021, 2020], tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "fake"
    assert manifest["nflreadpy_version"]
    assert [s["season"] for s in manifest["seasons"]] == [2020, 2021]
    for entry, record in zip(
        manifest["seasons"], sorted(records, key=lambda r: r.season), strict=True
    ):
        assert entry["rows"] == 4
        assert entry["null_id_rows"] == entry["duplicate_key_rows"] == 0
        assert entry["sha256"] == common.sha256_file(tmp_path / entry["file"]) == record.sha256


def test_manifest_counts_null_ids_and_kept_duplicates(tmp_path: Path) -> None:
    def fetch(season: int) -> pl.DataFrame:
        df = make_frame(season=season)
        nulls = df.head(2).with_columns(pl.lit(None, dtype=pl.String).alias("player_id"))
        return pl.concat([df, df.head(1), nulls])

    spec = dataclasses.replace(SPEC, fetch=fetch, duplicate_keys="warn")
    [record] = common.ingest_seasons(spec, [2021], tmp_path)
    assert (record.rows, record.null_id_rows, record.duplicate_key_rows) == (7, 2, 1)


def test_partial_run_keeps_other_seasons_in_manifest(tmp_path: Path) -> None:
    spec = with_fetch(SPEC)
    common.ingest_seasons(spec, [2020, 2021], tmp_path)

    common.ingest_seasons(spec, [2021], tmp_path, force=True)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [s["season"] for s in manifest["seasons"]] == [2020, 2021]


def test_ingest_does_not_write_invalid_data(tmp_path: Path) -> None:
    spec = dataclasses.replace(SPEC, fetch=lambda season: make_frame(season=season + 1))
    with pytest.raises(common.DataValidationError):
        common.ingest_seasons(spec, [2021], tmp_path)
    assert not common.season_path(tmp_path, 2021).exists()


def test_ingest_defaults_to_spec_out_dir(tmp_path: Path) -> None:
    spec = dataclasses.replace(with_fetch(SPEC), default_out_dir=tmp_path / "raw")
    common.ingest_seasons(spec, [2021])
    assert common.season_path(tmp_path / "raw", 2021).exists()


def parse(argv: list[str]) -> argparse.Namespace:
    return common.parse_season_args(common.build_parser(None, Path("out")), argv)


@pytest.mark.parametrize(
    "argv", [["--start-season", "2026", "--end-season", "2025"], ["--start-season", "1990"]]
)
def test_parse_args_rejects_bad_ranges(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse(argv)


def test_parse_args_defaults_exclude_holdout_season() -> None:
    args = parse([])
    assert (args.start_season, args.end_season) == (2018, 2025)
    assert args.out_dir == Path("out")


@pytest.mark.parametrize(
    ("doc", "expected"),
    [("Load things.\n\nMore.", "Load things."), ("\n  Lead.\n", "Lead."), (None, None)],
)
def test_first_line(doc: str | None, expected: str | None) -> None:
    assert common.first_line(doc) == expected
