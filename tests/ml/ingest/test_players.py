import json
import logging
from pathlib import Path

import polars as pl
import pytest

from ml.ingest import common
from ml.ingest import players as pl_ids


def make_players() -> pl.DataFrame:
    n = 3
    df = pl.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000002", "00-0000003"],
            "display_name": ["A", "B", "C"],
            "pfr_id": ["AaaA00", "BbbB00", None],
            **{c: [None] * n for c in pl_ids.ID_COLUMNS[2:]},
            # Present-day fields that must not reach the stored snapshot.
            "latest_team": ["AAA", "BBB", "CCC"],
            "status": ["ACT", "RES", "CUT"],
        },
        schema_overrides={c: pl.String for c in pl_ids.ID_COLUMNS},
    )
    return df


def test_validate_keeps_only_ids_and_name() -> None:
    out = pl_ids.validate_players(make_players())
    assert out.columns == pl_ids.KEPT_COLUMNS
    assert not {"latest_team", "status", "position"} & set(out.columns)


def test_validate_rejects_null_gsis_id() -> None:
    df = make_players().with_columns(pl.lit(None, dtype=pl.String).alias("gsis_id"))
    with pytest.raises(common.DataValidationError, match="null gsis_id"):
        pl_ids.validate_players(df)


@pytest.mark.parametrize("col", ["gsis_id", "pfr_id"])
def test_validate_rejects_duplicate_join_ids(col: str) -> None:
    df = make_players().with_columns(pl.lit("same").alias(col))
    with pytest.raises(common.DataValidationError, match=f"duplicate {col}"):
        pl_ids.validate_players(df)


def test_validate_rejects_missing_id_column() -> None:
    with pytest.raises(common.DataValidationError, match="missing columns"):
        pl_ids.validate_players(make_players().drop("pfr_id"))


def test_snap_coverage_counts_only_offensive_snaps() -> None:
    players = pl_ids.validate_players(make_players())
    snaps = pl.DataFrame(
        {
            "pfr_player_id": ["AaaA00", "BbbB00", "Unknown0", "Defense0"],
            "offense_snaps": [50.0, 20.0, 5.0, 0.0],
        }
    )
    assert pl_ids.snap_coverage(players, snaps) == pytest.approx(2 / 3)
    assert pl_ids.snap_coverage(players, snaps.clear()) == 1.0


def test_ingest_writes_snapshot_manifest_and_skips_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[int] = []

    def fake_fetch() -> pl.DataFrame:
        calls.append(1)
        return make_players()

    monkeypatch.setattr(pl_ids, "fetch_players", fake_fetch)
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    pl.DataFrame({"pfr_player_id": ["Unknown0"], "offense_snaps": [10.0]}).write_parquet(
        snap_dir / "season=2021.parquet"
    )

    with caplog.at_level(logging.INFO):
        path = pl_ids.ingest_players(tmp_path / "players", snap_dir)
    assert pl.read_parquet(path).height == 3
    manifest = json.loads((tmp_path / "players" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == common.sha256_file(path)
    assert manifest["columns"] == pl_ids.KEPT_COLUMNS
    assert any(r.levelno == logging.WARNING and "season=2021" in r.message for r in caplog.records)

    pl_ids.ingest_players(tmp_path / "players", snap_dir)
    assert len(calls) == 1
    pl_ids.ingest_players(tmp_path / "players", snap_dir, force=True)
    assert len(calls) == 2


@pytest.mark.network
def test_fetch_real_players_and_map_real_snaps() -> None:
    from ml.ingest import snap_counts as sc

    players = pl_ids.validate_players(pl_ids.fetch_players())
    assert players.height > 20_000
    assert pl_ids.snap_coverage(players, sc.fetch_snap_counts(2023)) > pl_ids.MIN_SNAP_COVERAGE
