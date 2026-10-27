import pytest

from backend.app.parsing.match import match_player
from backend.app.parsing.players import PlayerIndex, build_player_index
from backend.app.parsing.teams import canonical_team
from tests.backend.parsing.frames import extracted, rosters


@pytest.fixture(scope="module")
def index() -> PlayerIndex:
    return build_player_index(rosters(), season=2026)


@pytest.mark.parametrize(
    ("name", "position", "team", "player_id"),
    [
        ("Justin Jefferson", "WR", "MIN", "00-001"),
        ("Marvin Harrison", "WR", "ARI", "00-002"),  # suffix dropped on the screenshot
        ("AJ Brown", "WR", None, "00-003"),  # punctuation dropped
        ("J. Jefferson", "WR", None, "00-001"),  # abbreviated first name
        ("Amon-Ra St. Brown", "WR", "DET", "00-008"),
        ("Josh Allen", "QB", "BUF", "00-004"),  # the linebacker isn't a candidate
        ("Davante Adams", "WR", "LV", "00-011"),  # stale team on screenshot still matches
        ("Justin Jeferson", "WR", "MIN", "00-001"),  # one-letter typo
    ],
)
def test_matches(
    index: PlayerIndex, name: str, position: str, team: str | None, player_id: str
) -> None:
    result = match_player(extracted(name, position, team), index)
    assert result.status == "matched"
    assert result.player_id == player_id


def test_same_name_is_resolved_by_team(index: PlayerIndex) -> None:
    result = match_player(extracted("Mike Williams", "WR", "PIT"), index)
    assert (result.status, result.player_id) == ("matched", "00-007")


def test_same_name_without_team_is_ambiguous(index: PlayerIndex) -> None:
    result = match_player(extracted("Mike Williams", "WR"), index)
    assert result.status == "ambiguous"
    assert result.player_id is None
    assert {c.player_id for c in result.candidates} >= {"00-006", "00-007"}


def test_team_bonus_does_not_lift_a_weak_name_into_a_match(index: PlayerIndex) -> None:
    # "Justin T" is not "Justin Tucker"; a matching team must not turn it into a confident match.
    result = match_player(extracted("Justin T", "K", "BAL"), index)
    assert result.status != "matched"


def test_shared_surname_alone_is_not_a_match(index: PlayerIndex) -> None:
    # A player missing from the index must not snap onto a teammate-of-surname.
    result = match_player(extracted("Kyle Jefferson", "WR"), index)
    assert result.status == "unmatched"
    assert [c.player_id for c in result.candidates] == ["00-001"]  # still offered for review


def test_unknown_player_is_unmatched(index: PlayerIndex) -> None:
    result = match_player(extracted("Zxqv Plorbin", "WR"), index)
    assert result.status == "unmatched"
    assert result.candidates == []


def test_initial_shared_by_two_players_is_ambiguous() -> None:
    index = build_player_index(
        rosters(
            [
                ("00-1", 1, "Justin Jefferson", "WR", "MIN"),
                ("00-2", 1, "Jordan Jefferson", "WR", "NYG"),
            ]
        ),
        season=2026,
    )
    assert match_player(extracted("J. Jefferson", "WR"), index).status == "ambiguous"


def test_unknown_position_searches_all_players(index: PlayerIndex) -> None:
    result = match_player(extracted("Sam LaPorta", None, "DET"), index)
    assert (result.status, result.player_id) == ("matched", "00-012")


@pytest.mark.parametrize(
    ("name", "team"),
    [("Vikings D/ST", None), ("Minnesota Vikings", None), ("Vikings DEF", "MIN"), ("MIN", None)],
)
def test_defenses(index: PlayerIndex, name: str, team: str | None) -> None:
    result = match_player(extracted(name, "DST", team), index)
    assert (result.status, result.player_id) == ("matched", "DST:MIN")


def test_ambiguous_defense_location(index: PlayerIndex) -> None:
    # "New York" alone could be the Giants or the Jets.
    assert match_player(extracted("New York", "DST"), index).status != "matched"


@pytest.mark.parametrize(
    ("abbr", "expected"),
    [("WSH", "WAS"), ("lar", "LA"), ("JAC", "JAX"), (" kc ", "KC"), ("XYZ", None), (None, None)],
)
def test_canonical_team(abbr: str | None, expected: str | None) -> None:
    assert canonical_team(abbr) == expected
