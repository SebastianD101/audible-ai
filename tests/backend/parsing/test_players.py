from pathlib import Path

import pytest

from backend.app.parsing.players import build_player_index, load_player_index, normalize_name
from tests.backend.parsing.frames import rosters


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Marvin Harrison Jr.", "marvin harrison"),
        ("Kenneth Walker III", "kenneth walker"),
        ("A.J. Brown", "aj brown"),
        ("AJ Brown", "aj brown"),
        ("Amon-Ra St. Brown", "amon ra st brown"),
        ("Ja'Marr Chase", "jamarr chase"),
        ("  JOSH   ALLEN ", "josh allen"),
        ("José Núñez", "jose nunez"),
        ("V", "v"),  # a lone token is never stripped as a suffix
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_index_keeps_fantasy_positions_and_latest_week() -> None:
    index = build_player_index(rosters(), season=2026)
    by_id = {p.player_id: p for p in index.players}
    assert "00-005" not in by_id  # the linebacker Josh Allen
    assert by_id["00-011"].team == "LA"  # Davante Adams after the trade
    assert len([p for p in index.players if p.player_id == "00-011"]) == 1


def test_duplicate_gsis_rows_collapse_to_one_entry() -> None:
    # Upstream reused one gsis_id for two players in 2019; the index must not double-count it.
    rows = [
        ("00-099", 2, "Quinnen Williams", "QB", "NYJ"),
        ("00-099", 2, "Isaiah Searight", "QB", "NYG"),
    ]
    index = build_player_index(rosters(rows), season=2026)
    assert [p.player_id for p in index.players if not p.player_id.startswith("DST:")] == ["00-099"]


def test_index_adds_one_defense_per_team() -> None:
    index = build_player_index(rosters([]), season=2026)
    defenses = index.by_position("DST")
    assert len(defenses) == 32
    vikings = next(d for d in defenses if d.player_id == "DST:MIN")
    assert vikings.key == "vikings"
    assert "minnesota vikings" in vikings.aliases


def test_unknown_position_pool_excludes_defenses() -> None:
    index = build_player_index(rosters(), season=2026)
    assert all(p.position != "DST" for p in index.by_position(None))


def test_load_missing_season_explains_how_to_ingest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"ml.ingest.rosters_weekly"):
        load_player_index(2026, rosters_dir=tmp_path)


def test_load_reads_season_parquet(tmp_path: Path) -> None:
    rosters().write_parquet(tmp_path / "season=2026.parquet")
    assert load_player_index(2026, rosters_dir=tmp_path).season == 2026
