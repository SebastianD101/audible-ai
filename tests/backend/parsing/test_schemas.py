import pytest
from pydantic import ValidationError

from backend.app.parsing.schemas import ExtractedPlayer, ExtractedRoster


def player(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "Justin Jefferson",
        "position": "WR",
        "nfl_team": "MIN",
        "slot": "WR",
        "injury_tag": None,
        "legibility": "clear",
        "note": None,
    }
    return base | overrides


def test_roster_round_trips_through_json() -> None:
    roster = ExtractedRoster.model_validate(
        {"platform": "sleeper", "players": [player()], "warnings": []}
    )
    assert ExtractedRoster.model_validate_json(roster.model_dump_json()) == roster


@pytest.mark.parametrize("bad", [{"position": "LB"}, {"legibility": "unsure"}, {"name": None}])
def test_rejects_values_outside_the_schema(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExtractedPlayer.model_validate(player(**bad))


def test_unknown_position_is_null_not_guessed() -> None:
    assert ExtractedPlayer.model_validate(player(position=None)).position is None
