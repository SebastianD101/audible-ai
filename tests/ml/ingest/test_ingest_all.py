import dataclasses
from pathlib import Path

import polars as pl
import pytest

from ml.ingest import __main__ as ingest_all
from ml.ingest import common
from tests.ml.ingest.test_common import SPEC, make_frame


def spec(name: str, out_dir: Path, *, broken: bool = False) -> common.DatasetSpec:
    def fetch(season: int) -> pl.DataFrame:
        if broken:
            raise ConnectionError("nflverse unreachable")
        return make_frame(season)

    return dataclasses.replace(SPEC, name=name, fetch=fetch, default_out_dir=out_dir / name)


def test_one_failure_does_not_stop_the_others(tmp_path: Path) -> None:
    crosswalk_calls: list[bool] = []

    def crosswalk(*, force: bool) -> None:
        crosswalk_calls.append(force)

    specs = [spec("a", tmp_path), spec("b", tmp_path, broken=True), spec("c", tmp_path)]
    failed = ingest_all.run_all([2021], specs=specs, ingest_crosswalk=crosswalk)

    assert failed == ["b"]
    assert common.season_path(tmp_path / "a", 2021).exists()
    assert common.season_path(tmp_path / "c", 2021).exists()
    assert crosswalk_calls == [False]


def test_crosswalk_failure_is_reported(tmp_path: Path) -> None:
    def crosswalk(*, force: bool) -> None:
        raise ConnectionError("down")

    assert ingest_all.run_all([2021], specs=[spec("a", tmp_path)], ingest_crosswalk=crosswalk) == [
        "players"
    ]


def test_main_exits_nonzero_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest_all, "run_all", lambda seasons, force: ["schedules"])
    with pytest.raises(SystemExit) as exc:
        ingest_all.main([])
    assert exc.value.code == 1


def test_default_specs_cover_every_dataset_with_distinct_dirs() -> None:
    names = [s.name for s in ingest_all.SPECS]
    assert names == ["player_stats", "schedules", "rosters_weekly", "injuries", "snap_counts"]
    assert len({s.default_out_dir for s in ingest_all.SPECS}) == len(names)


def test_parse_args_defaults_exclude_holdout_season() -> None:
    args = ingest_all.parse_args([])
    assert (args.start_season, args.end_season, args.force) == (2018, 2025, False)
