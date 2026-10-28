import polars as pl
import pytest

from backend.app.parsing.players import PlayerIndex, build_player_index, load_player_index
from backend.app.projections import ProjectionTable, load_projections
from backend.app.sleeper.importer import (
    NotInLeagueError,
    free_agent_rows,
    import_league,
    league_settings,
    scoring_summary,
    scoring_warnings,
)
from backend.app.sleeper.schemas import SleeperLeague, SleeperRoster
from tests.backend.api.world import SEASON, write_world
from tests.backend.sleeper.fake import OTHER_ID, USER_ID, league, rosters, s

World = tuple[PlayerIndex, ProjectionTable]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    rosters_dir, projections_dir = write_world(tmp_path_factory.mktemp("world"))
    return load_player_index(SEASON, rosters_dir), load_projections(projections_dir)


def parsed(**overrides: object) -> tuple[SleeperLeague, list[SleeperRoster]]:
    return (
        SleeperLeague.model_validate(league(**overrides)),
        [SleeperRoster.model_validate(r) for r in rosters()],
    )


def test_roster_rows_follow_sleeper_slots(world: World) -> None:
    index, table = world
    lg, teams = parsed()
    result = import_league(lg, teams, USER_ID, index, table)
    rows = [(p.extracted.slot, p.match.player_id) for p in result.roster.players]
    assert rows == [
        ("QB", "qb1"), ("RB", "rb1"), ("RB", "rb2"), ("WR", "wr1"), ("WR", "wr3"),
        ("TE", "te1"), ("FLEX", "rb3"), ("W/R", "wr2"),  # the empty K slot is skipped
        ("DEF", "DST:BUF"), ("BN", "rb4"), ("BN", None), ("IR", "wr4"),
    ]  # fmt: skip
    assert result.roster.model == "sleeper-api"
    assert result.roster.platform == "sleeper"


def test_matched_rows_are_clear_and_unmapped_rows_are_flagged(world: World) -> None:
    index, table = world
    lg, teams = parsed()
    players = import_league(lg, teams, USER_ID, index, table).roster.players
    unmapped = [p for p in players if p.match.player_id is None]
    assert [p.extracted.name for p in unmapped] == ["Sleeper player 99999"]
    assert all(p.needs_review == (p.match.player_id is None) for p in players)


def test_co_owner_gets_their_team(world: World) -> None:
    index, table = world
    lg, teams = parsed()
    result = import_league(lg, teams, USER_ID + "0", index, table)
    assert [p.match.player_id for p in result.roster.players] == ["qb2", "te2"]


def test_user_without_a_team_is_an_error(world: World) -> None:
    index, table = world
    lg, teams = parsed()
    with pytest.raises(NotInLeagueError):
        import_league(lg, teams, "333", index, table)


def test_free_agents_skip_rostered_and_unprojected_players(world: World) -> None:
    index, table = world
    _, teams = parsed()
    fa = free_agent_rows(teams, table, index)
    ids = [p.match.player_id for p in fa.players]
    assert ids == ["qb3", "rb5", "wr5", "te3"]
    assert "wr6" not in ids  # on bye: no projection
    assert all(not p.needs_review for p in fa.players)
    assert "1 rostered player" in fa.warnings[0]


def test_free_agents_keep_the_top_n_per_position(world: World) -> None:
    index, table = world
    empty = [SleeperRoster(roster_id=1, owner_id=OTHER_ID)]
    fa = free_agent_rows(empty, table, index, per_position=2)
    ids = [p.match.player_id for p in fa.players]
    assert ids == ["qb1", "qb2", "rb1", "rb2", "wr1", "wr3", "te1", "te3"]
    assert fa.warnings == []


def test_settings_map_sleeper_slots() -> None:
    settings, warnings = league_settings(
        ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "WRRB_FLEX", "SUPER_FLEX", "K", "DEF"]
        + ["BN"] * 6 + ["IR"]
    )  # fmt: skip
    assert settings.slots == {
        "QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "WRRB_FLEX": 1, "SUPERFLEX": 1,
    }  # fmt: skip
    assert warnings == []


def test_lone_rec_flex_becomes_a_wr_te_flex() -> None:
    settings, warnings = league_settings(["QB", "WR", "TE", "REC_FLEX"])
    assert settings.slots["FLEX"] == 1
    assert settings.flex_positions == {"WR", "TE"}
    assert warnings == []


def test_rec_flex_beside_another_flex_is_approximated_with_a_warning() -> None:
    settings, warnings = league_settings(["QB", "WR", "FLEX", "REC_FLEX"])
    assert settings.slots["FLEX"] == 2
    assert settings.flex_positions == {"RB", "WR", "TE"}
    assert "W/T flex" in warnings[0]


def test_idp_slots_are_ignored_with_a_warning() -> None:
    _, warnings = league_settings(["QB", "RB", "DL", "LB", "IDP_FLEX"])
    assert warnings == ["Slots DL, IDP_FLEX, LB aren't projected and are ignored."]


def test_league_with_no_projected_slots_is_rejected() -> None:
    with pytest.raises(ValueError, match="slot"):
        league_settings(["K", "DEF", "BN"])


@pytest.mark.parametrize(
    ("scoring", "label", "fragments"),
    [
        ({"rec": 1.0, "pass_td": 4.0}, "PPR", []),
        ({"rec": 0.5}, "Half PPR", ["pass catchers are overvalued"]),
        ({"rec": 0.0}, "Standard", ["overvalued"]),
        ({"rec": 1.0, "bonus_rec_te": 0.5}, "PPR", ["tight ends are undervalued"]),
        ({"rec": 1.0, "pass_td": 6.0}, "PPR", ["quarterbacks are undervalued"]),
    ],
)
def test_scoring_label_and_warnings(
    scoring: dict[str, float], label: str, fragments: list[str]
) -> None:
    assert scoring_summary(scoring) == label
    warnings = scoring_warnings(scoring)
    assert len(warnings) == len(fragments)
    assert all(f in w for f, w in zip(fragments, warnings, strict=True))


def test_sleeper_team_codes_resolve_to_defenses(world: World) -> None:
    index, table = world
    lg, teams = parsed()
    teams[0].starters = s("qb1", "0", "0", "0", "0", "0", "0", "0", "0", "LAR")
    rows = import_league(lg, teams, USER_ID, index, table).roster.players
    assert rows[1].match.player_id == "DST:LA"  # Sleeper's LAR is nflverse's LA


def test_index_keeps_the_last_known_sleeper_id() -> None:
    frame = pl.DataFrame(
        {
            "season": [SEASON, SEASON],
            "week": [1, 2],
            "gsis_id": ["g1", "g1"],
            "full_name": ["Some Back", "Some Back"],
            "position": ["RB", "RB"],
            "team": ["NE", "NE"],
            "sleeper_id": ["4321", None],
        }
    )
    assert build_player_index(frame, SEASON).by_sleeper_id["4321"].player_id == "g1"
