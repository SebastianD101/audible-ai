import polars as pl
import pytest

from ml.features import load


def players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gsis_id": ["00-1", "00-2"],
            "display_name": ["Quinnen Williams", "Ann Back"],
            "pfr_id": ["WillQu00", "BackAn00"],
        }
    )


def test_clean_stats_keeps_regular_season_known_players_and_normalizes_teams() -> None:
    raw = pl.DataFrame(
        {
            "player_id": ["00-1", None, "00-2"],
            "player_display_name": ["A", "?", "B"],
            "position": ["WR", "WR", "RB"],
            "season": [2019] * 3,
            "week": [1, 1, 19],
            "season_type": ["REG", "REG", "POST"],
            "game_id": ["2019_01_DEN_OAK"] * 3,
            "team": ["LV"] * 3,
            "opponent_team": ["OAK"] * 3,
            **{c: [1.0] * 3 for c in load.STAT_COLUMNS},
        }
    )
    out = load.clean_stats(raw)
    assert out["player_id"].to_list() == ["00-1"]
    assert out.row(0, named=True)["opponent"] == "LV"
    assert out.row(0, named=True)["game_id"] == "2019_01_DEN_OAK"


def schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2019, 2019],
            "week": [1, 19],
            "game_type": ["REG", "WC"],
            "game_id": ["2019_01_DEN_OAK", "2019_19_BUF_HOU"],
            "home_team": ["OAK", "HOU"],
            "away_team": ["DEN", "BUF"],
            "home_score": [24, 22],
            "away_score": [16, 19],
            "home_rest": [7, 7],
            "away_rest": [6, 7],
            "div_game": [1, 0],
            "roof": ["outdoors", "closed"],
        }
    )


def test_team_games_gives_each_side_its_own_perspective() -> None:
    out = load.team_games(schedule())
    assert out.height == 2
    lv = out.filter(pl.col("team") == "LV").row(0, named=True)
    den = out.filter(pl.col("team") == "DEN").row(0, named=True)
    assert (lv["opponent"], lv["is_home"], lv["points_for"], lv["points_against"]) == (
        "DEN",
        True,
        24,
        16,
    )
    assert (den["is_home"], den["rest"], den["points_for"]) == (False, 6, 16)


def test_clean_rosters_resolves_upstream_id_collision_by_name() -> None:
    raw = pl.DataFrame(
        {
            "gsis_id": ["00-1", "00-1", "00-2", "00-3"],
            "full_name": ["Isaiah Searight", "Quinnen Williams", "Ann Back", "Kicker"],
            "season": [2019] * 4,
            "week": [2] * 4,
            "game_type": ["REG"] * 4,
            "team": ["NYG", "NYJ", "OAK", "LV"],
            "position": ["WR", "WR", "RB", "K"],
            "status": ["RES", "ACT", "ACT", "ACT"],
        }
    )
    out = load.clean_rosters(raw, players())
    assert out.select("player_id", "name", "team").rows() == [
        ("00-1", "Quinnen Williams", "NYJ"),
        ("00-2", "Ann Back", "LV"),
    ]


def test_clean_injuries_keeps_most_severe_and_nulls_blank_practice() -> None:
    raw = pl.DataFrame(
        {
            "gsis_id": ["00-1", "00-1", "00-2"],
            "season": [2024] * 3,
            "week": [15] * 3,
            "game_type": ["REG"] * 3,
            "team": ["HOU"] * 3,
            "report_status": ["Questionable", "Out", None],
            "practice_status": ["\n    ", "\n    ", "Full Participation in Practice"],
        }
    )
    out = load.clean_injuries(raw)
    assert out.select("player_id", "report_status", "severity", "practice_status").rows() == [
        ("00-1", "Out", 3, None),
        ("00-2", None, None, "Full Participation in Practice"),
    ]


def test_clean_snaps_maps_pfr_ids_and_drops_unmapped() -> None:
    raw = pl.DataFrame(
        {
            "pfr_player_id": ["WillQu00", "Nobody00"],
            "season": [2019, 2019],
            "week": [1, 1],
            "game_type": ["REG", "REG"],
            "game_id": ["g", "g"],
            "team": ["OAK", "OAK"],
            "offense_snaps": [40.0, 10.0],
            "offense_pct": [0.6, 0.15],
        }
    )
    out = load.clean_snaps(raw, players())
    assert out.select("player_id", "team", "offense_pct").rows() == [("00-1", "LV", 0.6)]


@pytest.mark.parametrize(("raw", "expected"), [(" x ", "x"), ("\n  ", None), (None, None)])
def test_blank_to_null(raw: str | None, expected: str | None) -> None:
    df = pl.DataFrame({"s": [raw]}, schema={"s": pl.String})
    assert df.select(load.blank_to_null("s")).item() == expected
