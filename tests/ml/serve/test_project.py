import dataclasses
import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml.features.asof import Week, before_expr
from ml.features.build import build_season
from ml.features.load import RawData
from ml.features.spine import OUTCOME_COLUMNS
from ml.models.lgbm import LgbmParams
from ml.serve import project
from ml.serve.project import StaleDataError, check_ready, project_week, training_rows
from tests.ml.features import synthetic as syn
from tests.ml.features.lookahead import corrupt_future

TARGET = Week(2023, 5)
# Small trees so the tiny synthetic league can actually move the projections.
PARAMS = LgbmParams(n_estimators=20, num_leaves=4, min_data_in_leaf=2)


@pytest.fixture(scope="module")
def past() -> pl.DataFrame:
    return build_season(syn.world(), 2022)


def corrupt_from(world: RawData, as_of: Week) -> RawData:
    future = ~before_expr(as_of)
    games = world.games.with_columns(
        pl.when(future).then(pl.col(c) * -7 + 999).otherwise(pl.col(c)).alias(c)
        for c in OUTCOME_COLUMNS
    )
    return dataclasses.replace(
        world,
        stats=corrupt_future(world.stats, as_of, strings=False),
        snaps=corrupt_future(world.snaps, as_of, strings=False),
        rosters=corrupt_future(world.rosters, as_of, strings=False),
        injuries=corrupt_future(world.injuries, as_of, strings=False),
        games=games,
    )


def test_projects_every_spine_player_with_a_hold_value(past: pl.DataFrame) -> None:
    out, n_train = project_week(past, syn.world(), TARGET, PARAMS)
    assert out.height == out["player_id"].n_unique() == 5
    assert out.columns == [*project.OUTPUT_COLUMNS, "projection", "hold_value"]
    assert (out["projection"] >= 0).all()
    assert (out["season"] == 2023).all() and (out["week"] == 5).all()
    assert n_train > past.height  # this season's earlier weeks were added
    qb = out.filter(pl.col("player_id") == "qb1")
    assert qb["hold_value"][0] == pytest.approx(20.0)  # scores 20 every game


def test_projections_ignore_everything_from_the_target_week_on(past: pl.DataFrame) -> None:
    clean, _ = project_week(past, syn.world(), TARGET, PARAMS)
    assert clean["projection"].n_unique() > 1, "constant projections would pass vacuously"
    leaked, _ = project_week(past, corrupt_from(syn.world(), TARGET), TARGET, PARAMS)
    assert_frame_equal(clean, leaked, check_row_order=False)


def test_training_rows_precede_the_target_week(past: pl.DataFrame) -> None:
    future_past = pl.concat([past, past.with_columns(pl.lit(2024, dtype=pl.Int32).alias("season"))])
    train = training_rows(future_past, syn.world(), TARGET)
    assert train.filter(~before_expr(TARGET)).is_empty()
    assert set(train.filter(pl.col("season") == 2023)["week"]) == {2, 3, 4}


def test_unfinished_previous_week_is_refused() -> None:
    world = syn.world()
    games = world.games.with_columns(
        pl.when((pl.col("season") == 2023) & (pl.col("week") == 4))
        .then(None)
        .otherwise(pl.col("points_for"))
        .alias("points_for")
    )
    with pytest.raises(StaleDataError, match="no final score"):
        check_ready(games, TARGET)


def test_week_off_the_schedule_is_refused() -> None:
    with pytest.raises(StaleDataError, match="not on the schedule"):
        check_ready(syn.world().games, Week(2023, 9))


def test_write_projections_records_version_and_provenance(tmp_path: Path) -> None:
    df = pl.DataFrame({"player_id": ["a"], "projection": [1.0]})
    path, manifest = project.write_projections(df, tmp_path, TARGET, {"git_commit": "abc"})
    assert path == tmp_path / "season=2023" / "week=05.parquet"
    on_disk = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["version"] == manifest["sha256"][:12]
    assert manifest["git_commit"] == "abc"


@pytest.mark.parametrize("argv", [["--season", "2025", "--week", "5"], ["--week", "1"]])
def test_cli_rejects_past_seasons_and_week_one(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        project.parse_args(argv)


def schedule(final_through: int, started: int = 0, weeks: int = 6) -> pl.DataFrame:
    """Two teams' games for 2023: weeks up to ``final_through`` are scored, plus ``started``
    teams' games in the next week."""
    rows = [
        {"season": 2023, "week": w, "team": t, "points_for": 20 if w <= final_through else None}
        for w in range(1, weeks + 1)
        for t in ("AAA", "BBB")
    ]
    for row in rows:
        if row["week"] == final_through + 1 and row["team"] in ("AAA", "BBB")[:started]:
            row["points_for"] = 20
    return pl.DataFrame(rows, schema_overrides={"points_for": pl.Int64})


@pytest.mark.parametrize(
    ("final_through", "expected"), [(0, Week(2023, 1)), (4, Week(2023, 5)), (6, None)]
)
def test_next_target_week_is_the_first_unplayed(final_through: int, expected: Week | None) -> None:
    assert project.next_target_week(schedule(final_through), 2023) == expected


def test_auto_projects_the_next_week(tmp_path: Path) -> None:
    assert project.auto_target(schedule(4), 2023, tmp_path) == (Week(2023, 5), "")


@pytest.mark.parametrize(
    ("games", "reason"),
    [
        (schedule(6), "season is over"),
        (schedule(0), "week 1"),
        (schedule(4, started=1), "already kicked off"),
    ],
)
def test_auto_skips_with_a_reason(tmp_path: Path, games: pl.DataFrame, reason: str) -> None:
    target, why = project.auto_target(games, 2023, tmp_path)
    assert target is None
    assert reason in why


def test_auto_never_overwrites_a_published_week(tmp_path: Path) -> None:
    project.write_projections(pl.DataFrame({"a": [1]}), tmp_path, Week(2023, 5), {})
    assert project.auto_target(schedule(4), 2023, tmp_path) == (None, "2023-W05: already projected")


def test_cli_accepts_auto() -> None:
    assert project.parse_args(["--week", "auto"]).week == "auto"
    with pytest.raises(SystemExit):
        project.parse_args(["--week", "soon"])
