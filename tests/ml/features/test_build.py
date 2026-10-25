import dataclasses
import json
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from ml.features import build
from ml.features.asof import Week, before_expr
from ml.features.spine import LABEL_COLUMNS, OUTCOME_COLUMNS, SPINE_COLUMNS
from tests.ml.features import synthetic as syn
from tests.ml.features.lookahead import corrupt_future


def test_build_week_has_one_row_per_player_and_labels_last() -> None:
    df = build.build_week(syn.world(), Week(2023, 3))
    assert df.height == df.select("player_id").unique().height == 5
    assert df.columns[: len(SPINE_COLUMNS)] == SPINE_COLUMNS
    assert df.columns[-len(LABEL_COLUMNS) :] == LABEL_COLUMNS
    for col in ("ppr_last3", "dvp_adj", "implied_total_proxy", "missed_last_game"):
        assert col in df.columns


def test_feature_columns_exclude_ids_and_labels() -> None:
    cols = build.feature_columns(build.build_week(syn.world(), Week(2023, 3)))
    assert not set(cols) & set(SPINE_COLUMNS + LABEL_COLUMNS)
    assert not [c for c in cols if c.startswith("label")]
    assert "target_share_season" in cols


def test_built_features_ignore_everything_from_the_target_week_on() -> None:
    """End-to-end leakage test: corrupt all future inputs (and future scores) at once."""
    as_of = Week(2023, 3)
    world = syn.world()
    future = ~before_expr(as_of)
    games = world.games.with_columns(
        pl.when(future).then(pl.col(c) * -7 + 999).otherwise(pl.col(c)).alias(c)
        for c in OUTCOME_COLUMNS
    )
    corrupted = dataclasses.replace(
        world,
        stats=corrupt_future(world.stats, as_of, strings=False),
        snaps=corrupt_future(world.snaps, as_of, strings=False),
        rosters=corrupt_future(world.rosters, as_of, strings=False),
        injuries=corrupt_future(world.injuries, as_of, strings=False),
        games=games,
    )
    waiver = pl.exclude("^ko_.*$")
    clean = build.week_features(world, as_of).select(waiver)
    assert clean.height == 5
    assert_frame_equal(
        clean, build.week_features(corrupted, as_of).select(waiver), check_row_order=False
    )


def test_kickoff_columns_ignore_everything_after_the_target_week() -> None:
    """With the cutoff moved to kickoff, corrupting week W+1 onward changes nothing at all."""
    as_of = Week(2023, 3)
    after = Week(2023, 4)
    world = syn.world()
    corrupted = dataclasses.replace(
        world,
        stats=corrupt_future(world.stats, as_of, strings=False),
        snaps=corrupt_future(world.snaps, as_of, strings=False),
        rosters=corrupt_future(world.rosters, as_of, strings=False),
        injuries=corrupt_future(world.injuries, after, strings=False),
        lines=corrupt_future(world.lines, after, strings=False),
    )
    clean = build.week_features(world, as_of)
    assert {"ko_total_line", "ko_report_severity"} <= set(clean.columns)
    assert_frame_equal(clean, build.week_features(corrupted, as_of), check_row_order=False)


def test_build_season_skips_week_one_and_byes() -> None:
    df = build.build_season(syn.world(), 2023)
    assert sorted(df["week"].unique()) == [2, 3, 4, 5]
    assert set(df.filter(pl.col("week") == 4)["team"]) == {"CCC", "DDD"}


def test_write_features_writes_files_and_traceable_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    (raw_dir / "schedules").mkdir(parents=True)
    (raw_dir / "schedules" / "manifest.json").write_text("{}", encoding="utf-8")

    manifest = build.write_features(syn.world(), [2022, 2023], tmp_path / "out", raw_dir)

    on_disk = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert [s["season"] for s in on_disk["seasons"]] == [2022, 2023]
    assert on_disk["label_columns"] == LABEL_COLUMNS
    assert on_disk["raw_manifest_sha256"]["schedules/manifest.json"]
    assert on_disk["raw_manifest_sha256"]["player_stats/manifest.json"] is None
    assert on_disk["git_commit"]
    stored = pl.read_parquet(tmp_path / "out" / "season=2023.parquet")
    assert stored.height == on_disk["seasons"][1]["rows"]


def test_parse_args_defaults_exclude_holdout_season() -> None:
    args = build.parse_args([])
    assert (args.start_season, args.end_season) == (2018, 2025)
