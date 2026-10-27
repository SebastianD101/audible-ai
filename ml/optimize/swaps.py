"""Recommend add/drop swaps that raise next week's projected starting-lineup points.

The search is a small mixed-integer program (PuLP + CBC):

- binary ``keep[i]`` per droppable roster player, ``add[j]`` per free agent, and
  ``start[p, s]`` per player and eligible slot type
- adds equal drops (roster size is fixed), at most ``max_moves`` adds
- a player starts at most once and only if owned; each slot type holds at most its count
- objective: projected starter points, minus a small cost per move (so it never churns for zero
  gain), plus a tiny bonus for keeping higher-value players (so a bench drop removes the
  least valuable player, not an arbitrary one)

Free agents are pruned exactly before solving: with at most k moves, only the k best-projected
free agents at each position can be worth adding, because any other could be swapped for a
better one at the same position that isn't already added.

Projections cover one week. To keep that horizon from recommending "drop your injured star",
roster players who would start on ``hold_values`` (a longer-horizon value such as a season
average) are protected from drops, as are ``locked`` players and anyone without a projection
(byes, K/DST, unmatched names), whose value the model can't judge.
"""

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import pulp

from ml.optimize.league import LeagueSettings, Player
from ml.optimize.lineup import Lineup, best_lineup

MOVE_COST = 1e-2  # per add, in projected points
KEEP_BONUS = 1e-3  # times a player's value scaled to [0, 1]
DEFAULT_MIN_GAIN = 0.5  # recommend nothing unless the lineup gains at least this much


@dataclass(frozen=True)
class Recommendation:
    """The best move set found, and enough context to explain it."""

    adds: tuple[str, ...]
    drops: tuple[str, ...]
    before: Lineup
    after: Lineup
    projected_gain: float
    protected: frozenset[str] = field(default_factory=frozenset)
    unprojected: frozenset[str] = field(default_factory=frozenset)

    @property
    def moved(self) -> bool:
        """Whether any swap is recommended."""
        return bool(self.adds)


@dataclass(frozen=True)
class Swap:
    """One add/drop pair and the projected lineup gain it gives on its own."""

    add: str
    drop: str
    projected_gain: float


def top_free_agents(
    free_agents: Iterable[Player], projections: Mapping[str, float], per_position: int
) -> list[Player]:
    """The ``per_position`` best-projected free agents at each position (positive only)."""
    by_position: dict[str, list[Player]] = {}
    for p in sorted(
        (p for p in free_agents if projections.get(p.player_id, 0.0) > 0.0),
        key=lambda p: (-projections[p.player_id], p.player_id),
    ):
        bucket = by_position.setdefault(p.position, [])
        if len(bucket) < per_position:
            bucket.append(p)
    return [p for bucket in by_position.values() for p in bucket]


def protected_players(
    roster: Sequence[Player],
    projections: Mapping[str, float],
    settings: LeagueSettings,
    locked: Iterable[str],
    hold_values: Mapping[str, float] | None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return (protected, unprojected) roster IDs; neither set may be dropped."""
    unprojected = frozenset(p.player_id for p in roster if p.player_id not in projections)
    protected = set(locked) & {p.player_id for p in roster}
    if hold_values is not None:
        protected |= best_lineup(roster, hold_values, settings).player_ids
    return frozenset(protected), unprojected


def keep_values(
    players: Sequence[Player], projections: Mapping[str, float], hold: Mapping[str, float] | None
) -> dict[str, float]:
    """Each player's value scaled to [0, 1], for the keep-the-better-player tie-break."""
    source = hold if hold is not None else projections
    raw = {p.player_id: max(source.get(p.player_id, 0.0), 0.0) for p in players}
    top = max(raw.values(), default=0.0)
    return {pid: (v / top if top > 0 else 0.0) for pid, v in raw.items()}


def cbc_solver() -> pulp.LpSolver:
    """The CBC binary bundled with PuLP 3.x, without its PuLP-4 deprecation notice.

    PuLP 4 moves CBC to the ``pulp[cbc]`` extra and ``COIN_CMD``; switch when upgrading.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "PULP_CBC_CMD is deprecated", DeprecationWarning)
        return pulp.PULP_CBC_CMD(msg=False)


def solve_moves(
    roster: Sequence[Player],
    droppable: frozenset[str],
    candidates: Sequence[Player],
    projections: Mapping[str, float],
    settings: LeagueSettings,
    max_moves: int,
    values: Mapping[str, float],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Solve the MILP and return (adds, drops), each sorted.

    Variables are created in sorted order: set iteration order varies between processes, and
    a different model order can change which of several equal optima CBC returns.
    """
    prob = pulp.LpProblem("swaps", pulp.LpMaximize)

    def binary(name: str) -> pulp.LpVariable:
        return prob.add_variable(name, 0, 1, cat="Binary")

    keep = {pid: binary(f"keep_{n}") for n, pid in enumerate(sorted(droppable))}
    add = {p.player_id: binary(f"add_{n}") for n, p in enumerate(candidates)}

    def owned(pid: str) -> pulp.LpVariable | int:
        if pid in keep:
            return keep[pid]
        return add.get(pid, 1)

    start: dict[tuple[str, str], pulp.LpVariable] = {}
    for p in sorted([*roster, *candidates], key=lambda p: p.player_id):
        if projections.get(p.player_id, 0.0) <= 0.0:
            continue
        for slot, _ in settings.ordered_slots():
            if p.position in settings.eligible(slot):
                start[p.player_id, slot] = binary(f"start_{len(start)}")

    prob += (
        pulp.lpSum(projections[pid] * var for (pid, _), var in start.items())
        - MOVE_COST * pulp.lpSum(add.values())
        + KEEP_BONUS * pulp.lpSum(values.get(pid, 0.0) * var for pid, var in keep.items())
    )
    prob += pulp.lpSum(add.values()) == pulp.lpSum(1 - var for var in keep.values())
    prob += pulp.lpSum(add.values()) <= max_moves
    for pid in sorted({pid for pid, _ in start}):
        prob += pulp.lpSum(var for (q, _), var in start.items() if q == pid) <= owned(pid)
    for slot, count in settings.ordered_slots():
        prob += pulp.lpSum(var for (_, s), var in start.items() if s == slot) <= count

    prob.solve(cbc_solver())
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"swap optimizer did not converge: {pulp.LpStatus[prob.status]}")
    adds = tuple(sorted(pid for pid, var in add.items() if (var.value() or 0) > 0.5))
    drops = tuple(sorted(pid for pid, var in keep.items() if (var.value() or 0) < 0.5))
    return adds, drops


def recommend(
    roster: Sequence[Player],
    free_agents: Sequence[Player],
    projections: Mapping[str, float],
    settings: LeagueSettings,
    *,
    max_moves: int = 1,
    locked: Iterable[str] = (),
    hold_values: Mapping[str, float] | None = None,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> Recommendation:
    """Find up to ``max_moves`` add/drop swaps that most raise projected lineup points.

    Returns a no-move recommendation when the best gain found is below ``min_gain``.

    Raises:
        ValueError: if ``max_moves`` is negative.
        RuntimeError: if the solver fails.
    """
    if max_moves < 0:
        raise ValueError("max_moves must be >= 0")
    protected, unprojected = protected_players(roster, projections, settings, locked, hold_values)
    before = best_lineup(roster, projections, settings)
    no_move = Recommendation((), (), before, before, 0.0, protected, unprojected)
    droppable = frozenset(p.player_id for p in roster) - protected - unprojected
    on_roster = {p.player_id for p in roster}
    candidates = top_free_agents(
        (p for p in free_agents if p.player_id not in on_roster), projections, max_moves
    )
    if max_moves == 0 or not droppable or not candidates:
        return no_move

    values = keep_values(roster, projections, hold_values)
    adds, drops = solve_moves(
        roster, droppable, candidates, projections, settings, max_moves, values
    )
    by_id = {p.player_id: p for p in candidates}
    new_roster = [p for p in roster if p.player_id not in drops] + [by_id[a] for a in adds]
    after = best_lineup(new_roster, projections, settings)
    gain = after.points - before.points
    if not adds or gain < min_gain:
        return no_move
    return Recommendation(adds, drops, before, after, gain, protected, unprojected)


def rank_single_swaps(
    roster: Sequence[Player],
    free_agents: Sequence[Player],
    projections: Mapping[str, float],
    settings: LeagueSettings,
    *,
    n: int = 5,
    locked: Iterable[str] = (),
    hold_values: Mapping[str, float] | None = None,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> list[Swap]:
    """Top ``n`` single swaps (best drop per added player), for showing alternatives.

    Each candidate is scored by re-running the greedy lineup, which is exact and fast; the
    pool is pruned to the ``n`` best free agents per position, which is all a top-``n`` list
    can contain.
    """
    protected, unprojected = protected_players(roster, projections, settings, locked, hold_values)
    base = best_lineup(roster, projections, settings).points
    on_roster = {p.player_id for p in roster}
    values = keep_values(roster, projections, hold_values)
    # Lowest-value droppable first, so ties keep the better player.
    droppable = sorted(
        (p for p in roster if p.player_id not in protected | unprojected),
        key=lambda p: (values[p.player_id], p.player_id),
    )
    swaps = []
    for fa in top_free_agents(
        (p for p in free_agents if p.player_id not in on_roster), projections, n
    ):
        best: Swap | None = None
        for d in droppable:
            lineup = best_lineup([*(p for p in roster if p != d), fa], projections, settings)
            gain = lineup.points - base
            if best is None or gain > best.projected_gain + 1e-9:
                best = Swap(fa.player_id, d.player_id, gain)
        if best is not None and best.projected_gain >= min_gain:
            swaps.append(best)
    return sorted(swaps, key=lambda s: (-s.projected_gain, s.add))[:n]
