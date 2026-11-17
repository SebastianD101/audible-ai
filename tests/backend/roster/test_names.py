import pytest

from backend.app.roster.names import INITIAL_SCORE, name_score, split_ratio
from backend.app.roster.players import (
    IndexedPlayer,
    PlayerIndex,
    build_player_index,
    normalize_name,
)
from backend.app.roster.teams import canonical_team
from tests.backend.roster.frames import rosters


@pytest.fixture(scope="module")
def index() -> PlayerIndex:
    return build_player_index(rosters(), season=2026)


def player(index: PlayerIndex, player_id: str) -> IndexedPlayer:
    return next(p for p in index.by_position(None) if p.player_id == player_id)


def score(index: PlayerIndex, query: str, player_id: str) -> float:
    return name_score(normalize_name(query), player(index, player_id))


def test_exact_and_alias_names_score_one(index: PlayerIndex) -> None:
    assert score(index, "Justin Jefferson", "00-001") == 1.0
    assert score(index, "Marvin Harrison", "00-002") == 1.0  # suffix dropped
    assert score(index, "AJ Brown", "00-003") == 1.0  # punctuation dropped


def test_abbreviated_first_name(index: PlayerIndex) -> None:
    assert score(index, "J. Jefferson", "00-001") == INITIAL_SCORE
    assert score(index, "K. Jefferson", "00-001") < INITIAL_SCORE


def test_one_letter_typo_scores_high(index: PlayerIndex) -> None:
    assert score(index, "Justin Jeferson", "00-001") > 0.9


def test_shared_surname_alone_scores_low(index: PlayerIndex) -> None:
    assert score(index, "Kevin Jefferson", "00-001") < 0.8


def test_split_ratio_falls_back_for_one_word_names() -> None:
    assert split_ratio("adams", "adams") == 1.0
    assert split_ratio("davante adams", "davante adams") == 1.0


@pytest.mark.parametrize(
    ("abbr", "expected"),
    [("WSH", "WAS"), ("lar", "LA"), ("JAC", "JAX"), (" kc ", "KC"), ("XYZ", None), (None, None)],
)
def test_canonical_team(abbr: str | None, expected: str | None) -> None:
    assert canonical_team(abbr) == expected
