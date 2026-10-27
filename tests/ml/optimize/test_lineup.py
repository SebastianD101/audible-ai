import itertools

import numpy as np
import pytest
from pydantic import ValidationError

from ml.optimize.league import LeagueSettings, Player, SlotName
from ml.optimize.lineup import best_lineup

POSITIONS = ("QB", "RB", "WR", "TE")


def brute_force(players: list[Player], proj: dict[str, float], settings: LeagueSettings) -> float:
    """Best total over every assignment of players to slots (empty allowed)."""
    seats: list[SlotName] = [s for s, n in settings.ordered_slots() for _ in range(n)]
    best = 0.0
    options = [None, *players]
    for combo in itertools.product(options, repeat=len(seats)):
        chosen = [p for p in combo if p is not None]
        if len({p.player_id for p in chosen}) < len(chosen):
            continue
        if any(
            p is not None and p.position not in settings.eligible(seat)
            for seat, p in zip(seats, combo, strict=True)
        ):
            continue
        best = max(best, sum(max(proj[p.player_id], 0.0) for p in chosen))
    return best


def random_case(seed: int, n: int) -> tuple[list[Player], dict[str, float]]:
    rng = np.random.default_rng(seed)
    players = [Player(f"p{i}", str(rng.choice(POSITIONS))) for i in range(n)]
    proj = {p.player_id: float(rng.normal(10, 6)) for p in players}
    return players, proj


@pytest.mark.parametrize("seed", range(25))
def test_greedy_matches_brute_force_default_slots(seed: int) -> None:
    settings = LeagueSettings(slots={"QB": 1, "RB": 1, "WR": 1, "FLEX": 1})
    players, proj = random_case(seed, n=7)
    assert best_lineup(players, proj, settings).points == pytest.approx(
        brute_force(players, proj, settings)
    )


@pytest.mark.parametrize("seed", range(15))
def test_greedy_matches_brute_force_superflex(seed: int) -> None:
    settings = LeagueSettings(slots={"QB": 1, "TE": 1, "FLEX": 1, "SUPERFLEX": 1})
    players, proj = random_case(seed + 100, n=7)
    assert best_lineup(players, proj, settings).points == pytest.approx(
        brute_force(players, proj, settings)
    )


def test_flex_takes_best_remaining_eligible_player() -> None:
    players = [
        Player("qb1", "QB"),
        Player("qb2", "QB"),
        Player("rb1", "RB"),
        Player("rb2", "RB"),
        Player("rb3", "RB"),
        Player("wr1", "WR"),
        Player("wr2", "WR"),
        Player("te1", "TE"),
    ]
    proj = {"qb1": 20, "qb2": 30, "rb1": 15, "rb2": 9, "rb3": 12, "wr1": 14, "wr2": 8, "te1": 6}
    lineup = best_lineup(players, proj, LeagueSettings())
    slots = dict((pid, slot) for slot, pid in lineup.starters)
    assert slots == {"qb2": "QB", "rb1": "RB", "rb3": "RB", "wr1": "WR", "wr2": "WR",
                     "te1": "TE", "rb2": "FLEX"}  # fmt: skip
    assert lineup.points == 30 + 15 + 12 + 14 + 8 + 6 + 9


def test_unprojected_and_nonpositive_players_never_start() -> None:
    players = [Player("rb1", "RB"), Player("rb_bye", "RB"), Player("rb_zero", "RB")]
    lineup = best_lineup(players, {"rb1": 10.0, "rb_zero": 0.0}, LeagueSettings())
    assert lineup.player_ids == {"rb1"}  # other slots stay empty
    assert lineup.points == 10.0


def test_qb_cannot_fill_flex() -> None:
    lineup = best_lineup([Player("qb", "QB"), Player("qb2", "QB")], {"qb": 9, "qb2": 8},
                         LeagueSettings())  # fmt: skip
    assert lineup.player_ids == {"qb"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scoring": "standard"},
        {"slots": {"QB": 1, "K": 1}},
        {"slots": {"QB": 1, "DST": 1}},
        {"slots": {"QB": -1}},
        {"slots": {"QB": 0}},
        {"flex_positions": frozenset({"QB", "RB"})},
        {"flex_positions": frozenset()},
    ],
)
def test_settings_reject_unsupported_configs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LeagueSettings.model_validate(kwargs)


def test_k_slot_error_explains_why() -> None:
    with pytest.raises(ValidationError, match="not projected"):
        LeagueSettings.model_validate({"slots": {"QB": 1, "K": 1}})
