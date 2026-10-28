"""Best starting lineup for a set of players under projected points.

Greedy is exact here. Slot types are filled in ``SLOT_ORDER``, and each type's eligible
positions contain every earlier type's (QB ⊂ SUPERFLEX, RB ⊂ WRRB_FLEX ⊂ FLEX ⊂ SUPERFLEX,
...). Filling a dedicated slot with anyone but the best remaining player of that position
can't help: swapping in the better one frees a worse player for the later, broader slots, never
a better one. The same argument applies to each flex type before the next, broader one. Tests
check this against brute force.

A slot is left empty rather than filled with a non-positive projection, matching a manager who
would rather bench a zero than start it. Players without a projection (byes, K/DST, unmatched
names) never start.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ml.optimize.league import LeagueSettings, Player, SlotName


@dataclass(frozen=True)
class Lineup:
    """Chosen starters as (slot, player_id) pairs, and their total projected points."""

    starters: tuple[tuple[SlotName, str], ...]
    points: float

    @property
    def player_ids(self) -> frozenset[str]:
        """IDs of everyone starting."""
        return frozenset(pid for _, pid in self.starters)


def best_lineup(
    players: Iterable[Player], projections: Mapping[str, float], settings: LeagueSettings
) -> Lineup:
    """Return the highest-projected legal lineup from ``players``."""
    pool = sorted(
        (p for p in players if projections.get(p.player_id, 0.0) > 0.0),
        key=lambda p: (-projections[p.player_id], p.player_id),
    )
    used: set[str] = set()
    starters: list[tuple[SlotName, str]] = []
    for slot, count in settings.ordered_slots():
        eligible = settings.eligible(slot)
        picks = [p for p in pool if p.player_id not in used and p.position in eligible][:count]
        for p in picks:
            used.add(p.player_id)
            starters.append((slot, p.player_id))
    return Lineup(starters=tuple(starters), points=sum(projections[pid] for _, pid in starters))
