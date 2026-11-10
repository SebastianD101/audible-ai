import pytest
from pydantic import ValidationError

from backend.app.roster.schemas import ExtractedPlayer


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


def test_player_round_trips_through_json() -> None:
    row = ExtractedPlayer.model_validate(player())
    assert ExtractedPlayer.model_validate_json(row.model_dump_json()) == row


@pytest.mark.parametrize("bad", [{"position": "LB"}, {"legibility": "unsure"}, {"name": None}])
def test_rejects_values_outside_the_schema(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExtractedPlayer.model_validate(player(**bad))


def test_unknown_position_is_null_not_guessed() -> None:
    assert ExtractedPlayer.model_validate(player(position=None)).position is None
