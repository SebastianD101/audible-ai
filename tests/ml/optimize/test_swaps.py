import itertools

import numpy as np
import pytest

from ml.optimize.league import LeagueSettings, Player
from ml.optimize.lineup import best_lineup
from ml.optimize.swaps import rank_single_swaps, recommend

SETTINGS = LeagueSettings()
POSITIONS = ("QB", "RB", "WR", "TE")


def roster_and_pool() -> tuple[list[Player], list[Player], dict[str, float]]:
    roster = [
        Player("qb1", "QB"),
        Player("rb1", "RB"),
        Player("rb2", "RB"),
        Player("rb3", "RB"),
        Player("wr1", "WR"),
        Player("wr2", "WR"),
        Player("wr3", "WR"),
        Player("te1", "TE"),
    ]
    pool = [Player("fa_rb", "RB"), Player("fa_wr", "WR"), Player("fa_te", "TE")]
    proj = {
        "qb1": 20, "rb1": 15, "rb2": 12, "rb3": 4, "wr1": 14, "wr2": 11, "wr3": 3, "te1": 5,
        "fa_rb": 9, "fa_wr": 13, "fa_te": 10,
    }  # fmt: skip
    return roster, pool, {k: float(v) for k, v in proj.items()}


def brute_force_gain(
    roster: list[Player],
    pool: list[Player],
    proj: dict[str, float],
    droppable: set[str],
    k: int,
) -> float:
    base = best_lineup(roster, proj, SETTINGS).points
    best = 0.0
    drop_pool = [p for p in roster if p.player_id in droppable]
    for m in range(1, k + 1):
        for adds in itertools.combinations(pool, m):
            for drops in itertools.combinations(drop_pool, m):
                new = [p for p in roster if p not in drops] + list(adds)
                best = max(best, best_lineup(new, proj, SETTINGS).points - base)
    return best


def test_finds_the_obvious_upgrade() -> None:
    roster, pool, proj = roster_and_pool()
    rec = recommend(roster, pool, proj, SETTINGS)
    # Lineup: qb1, rb1, rb2, wr1, wr2, te1, FLEX rb3 (4) = 81. Adding fa_wr (13) starts it at
    # WR and pushes wr2 (11) to FLEX: +9, which beats fa_te over te1 (+5) or fa_rb in FLEX (+5).
    assert rec.adds == ("fa_wr",)
    assert rec.projected_gain == pytest.approx(9.0)
    assert rec.after.points - rec.before.points == pytest.approx(rec.projected_gain)


def test_drop_prefers_least_valuable_bench_player() -> None:
    roster, pool, proj = roster_and_pool()
    rec = recommend(roster, pool, proj, SETTINGS)
    # rb3 (4) and wr3 (3) are both benched after the move; wr3 is worth less.
    assert rec.drops == ("wr3",)


def test_two_moves() -> None:
    roster, pool, proj = roster_and_pool()
    rec = recommend(roster, pool, proj, SETTINGS, max_moves=2)
    assert set(rec.adds) == {"fa_te", "fa_wr"}  # +9 and +5
    assert set(rec.drops) == {"wr3", "rb3"}  # te1 (5) is worth more than either
    assert rec.projected_gain == pytest.approx(14.0)


def test_roster_size_is_preserved() -> None:
    roster, pool, proj = roster_and_pool()
    rec = recommend(roster, pool, proj, SETTINGS, max_moves=3)
    assert len(rec.adds) == len(rec.drops)


def test_locked_players_are_never_dropped() -> None:
    roster, pool, proj = roster_and_pool()
    rec = recommend(roster, pool, proj, SETTINGS, max_moves=3, locked={"wr3", "rb3", "te1"})
    assert not set(rec.drops) & {"wr3", "rb3", "te1"}


def test_no_move_below_min_gain() -> None:
    roster, pool, proj = roster_and_pool()
    rec = recommend(roster, pool, proj, SETTINGS, min_gain=10.0)  # best single move is +9
    assert not rec.moved
    assert rec.after == rec.before
    assert rec.projected_gain == 0.0


def test_no_move_when_free_agents_are_worse() -> None:
    roster, _, proj = roster_and_pool()
    pool = [Player("scrub", "WR")]
    rec = recommend(roster, pool, proj | {"scrub": 1.0}, SETTINGS)
    assert not rec.moved


def test_hold_values_protect_an_injured_star() -> None:
    roster, pool, proj = roster_and_pool()
    proj["rb1"] = 0.2  # ruled out this week
    hold = dict(proj) | {"rb1": 18.0}  # but a season-long starter
    unguarded = recommend(roster, pool, proj, SETTINGS, max_moves=3)
    guarded = recommend(roster, pool, proj, SETTINGS, max_moves=3, hold_values=hold)
    assert "rb1" in unguarded.drops or "rb1" not in guarded.drops  # guard can only help
    assert "rb1" in guarded.protected
    assert "rb1" not in guarded.drops


def test_unprojected_roster_players_are_kept_and_reported() -> None:
    roster, pool, proj = roster_and_pool()
    roster.append(Player("bye_wr", "WR"))  # on bye: no projection this week
    rec = recommend(roster, pool, proj, SETTINGS, max_moves=3)
    assert "bye_wr" in rec.unprojected
    assert "bye_wr" not in rec.drops


def test_free_agent_already_on_roster_is_ignored() -> None:
    roster, pool, proj = roster_and_pool()
    rec = recommend(roster, [*pool, Player("rb1", "RB")], proj, SETTINGS, max_moves=2)
    assert "rb1" not in rec.adds


def test_negative_max_moves_rejected() -> None:
    roster, pool, proj = roster_and_pool()
    with pytest.raises(ValueError, match="max_moves"):
        recommend(roster, pool, proj, SETTINGS, max_moves=-1)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("k", [1, 2])
def test_matches_brute_force(seed: int, k: int) -> None:
    rng = np.random.default_rng(seed)
    roster = [Player(f"r{i}", str(rng.choice(POSITIONS))) for i in range(8)]
    pool = [Player(f"f{i}", str(rng.choice(POSITIONS))) for i in range(6)]
    proj = {p.player_id: float(rng.gamma(2.0, 5.0)) for p in [*roster, *pool]}
    rec = recommend(roster, pool, proj, SETTINGS, max_moves=k, min_gain=0.0)
    expected = brute_force_gain(roster, pool, proj, {p.player_id for p in roster}, k)
    assert rec.projected_gain == pytest.approx(expected, abs=1e-6)


def test_rank_single_swaps_orders_by_gain_and_respects_guards() -> None:
    roster, pool, proj = roster_and_pool()
    swaps = rank_single_swaps(roster, pool, proj, SETTINGS, n=3, locked={"te1"})
    # fa_wr +9. fa_te starts at TE and the locked te1 (5) moves to FLEX over rb3 (4): +6.
    # fa_rb replaces rb3 in FLEX: +5.
    assert [s.add for s in swaps] == ["fa_wr", "fa_te", "fa_rb"]
    assert [s.projected_gain for s in swaps] == pytest.approx([9.0, 6.0, 5.0])
    assert all(s.drop != "te1" for s in swaps)
    assert swaps[0].drop == "wr3"  # least valuable bench player


def test_rank_single_swaps_best_entry_matches_recommend() -> None:
    rng = np.random.default_rng(7)
    roster = [Player(f"r{i}", str(rng.choice(POSITIONS))) for i in range(9)]
    pool = [Player(f"f{i}", str(rng.choice(POSITIONS))) for i in range(10)]
    proj = {p.player_id: float(rng.gamma(2.0, 5.0)) for p in [*roster, *pool]}
    swaps = rank_single_swaps(roster, pool, proj, SETTINGS, min_gain=0.0)
    rec = recommend(roster, pool, proj, SETTINGS, min_gain=0.0)
    assert swaps[0].projected_gain == pytest.approx(rec.projected_gain)
