"""Turn a reviewed roster and free-agent list into an add/drop recommendation.

Thin glue around ``ml.optimize``: resolve IDs to positions, pull this week's projections and
season-long hold values, run the optimizer, and attach display info, reasons, caveats, and the
backtest evidence the UI shows next to the number.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.parsing.players import PlayerIndex
from backend.app.projections import ProjectionsMeta, ProjectionTable
from backend.app.search import PlayerHit, to_hit
from ml.optimize.league import LeagueSettings, Player
from ml.optimize.lineup import Lineup
from ml.optimize.swaps import rank_single_swaps, recommend

MAX_ROSTER = 40
MAX_FREE_AGENTS = 400
N_ALTERNATIVES = 5


class RecommendationRequest(BaseModel):
    """A reviewed roster, the free agents available, and league settings."""

    roster: list[str] = Field(min_length=1, max_length=MAX_ROSTER)
    free_agents: list[str] = Field(default_factory=list, max_length=MAX_FREE_AGENTS)
    locked: list[str] = Field(default_factory=list, max_length=MAX_ROSTER)
    max_moves: int = Field(default=1, ge=1, le=3)
    settings: LeagueSettings = Field(default_factory=LeagueSettings)


class LineupSlot(BaseModel):
    """One starter."""

    slot: str
    player: PlayerHit


class LineupOut(BaseModel):
    """A starting lineup and its projected points."""

    starters: list[LineupSlot]
    points: float


class SwapOut(BaseModel):
    """An alternative single add/drop."""

    add: PlayerHit
    drop: PlayerHit
    projected_gain: float


ProtectionReason = Literal["locked", "no_projection", "season_starter"]


class Protection(BaseModel):
    """A roster player the optimizer would not drop, and why."""

    player: PlayerHit
    reason: ProtectionReason


class BacktestEvidence(BaseModel):
    """How recommendations like this one did historically (``docs/pickups.md``)."""

    realized_gain: float
    realized_gain_lo: float
    realized_gain_hi: float
    hit_rate: float
    roster_weeks: int
    weeks: int
    summary: str


class Recommendation(BaseModel):
    """The recommended moves and everything needed to explain them."""

    season: int
    week: int
    projections_version: str
    adds: list[PlayerHit]
    drops: list[PlayerHit]
    projected_gain: float
    before: LineupOut
    after: LineupOut
    alternatives: list[SwapOut]
    protected: list[Protection]
    caveats: list[str]
    evidence: BacktestEvidence | None


class UnknownPlayersError(ValueError):
    """Request IDs that aren't in this season's player index."""

    def __init__(self, ids: Sequence[str]) -> None:
        """Keep the offending IDs for the error response."""
        super().__init__(f"unknown player IDs: {', '.join(ids)}")
        self.ids = list(ids)


# Simulated-league pickup backtest, LightGBM-driven moves, 2020-2025 (docs/pickups.md).
EVIDENCE: dict[int, BacktestEvidence] = {
    k: BacktestEvidence(
        realized_gain=gain,
        realized_gain_lo=lo,
        realized_gain_hi=hi,
        hit_rate=hit,
        roster_weeks=5050,
        weeks=101,
        summary=(
            f"In a 2020-2025 backtest of {5050:,} simulated roster-weeks, following up to {k} "
            f"recommended move(s) gained {gain:.2f} real points per week on average (95% CI "
            f"{lo:.2f} to {hi:.2f}), and {hit:.0%} of moves made gained points. Single weeks "
            "are noisy: a recommended move often loses."
        ),
    )
    for k, gain, lo, hi, hit in [(1, 1.98, 1.27, 2.75, 0.61), (2, 2.46, 1.54, 3.46, 0.62)]
}


def unique(ids: Sequence[str]) -> list[str]:
    """``ids`` without duplicates, first occurrence kept."""
    return list(dict.fromkeys(ids))


def lineup_out(lineup: Lineup, hits: dict[str, PlayerHit]) -> LineupOut:
    """Attach display info to a lineup."""
    return LineupOut(
        starters=[LineupSlot(slot=s, player=hits[pid]) for s, pid in lineup.starters],
        points=round(lineup.points, 2),
    )


def caveats(meta: ProjectionsMeta, n_free_agents: int, protected: list[Protection]) -> list[str]:
    """Plain-language limits of this recommendation."""
    out = [
        f"Projections are for week {meta.week} only. Season-long value is used only to keep "
        "your season-long starters from being dropped.",
        f"Only the {n_free_agents} free agent(s) you listed were considered.",
        "Kickers and defenses are not projected, so they are never added, dropped, or counted "
        "in lineup points.",
    ]
    if any(p.reason == "no_projection" for p in protected):
        out.append(
            "Players without a projection this week (on bye, or not on an active roster) "
            "are kept as-is."
        )
    return out


def build_recommendation(
    request: RecommendationRequest, table: ProjectionTable, index: PlayerIndex
) -> Recommendation:
    """Run the optimizer on the request and explain the result.

    Raises:
        UnknownPlayersError: if any ID isn't in the player index.
    """
    by_id = {p.player_id: p for p in index.players}
    roster_ids, fa_ids = unique(request.roster), unique(request.free_agents)
    unknown = [i for i in unique([*roster_ids, *fa_ids, *request.locked]) if i not in by_id]
    if unknown:
        raise UnknownPlayersError(unknown)

    hits = {i: to_hit(by_id[i], table) for i in {*roster_ids, *fa_ids}}
    roster = [Player(i, by_id[i].position) for i in roster_ids]
    free_agents = [Player(i, by_id[i].position) for i in fa_ids if i not in set(roster_ids)]
    projections, holds = table.projections(), table.hold_values()
    locked = set(request.locked) & set(roster_ids)

    rec = recommend(
        roster,
        free_agents,
        projections,
        request.settings,
        max_moves=request.max_moves,
        locked=locked,
        hold_values=holds,
    )
    alternatives = rank_single_swaps(
        roster,
        free_agents,
        projections,
        request.settings,
        n=N_ALTERNATIVES,
        locked=locked,
        hold_values=holds,
    )

    def reason(pid: str) -> ProtectionReason:
        if pid in locked:
            return "locked"
        return "no_projection" if pid in rec.unprojected else "season_starter"

    protected = [
        Protection(player=hits[pid], reason=reason(pid))
        for pid in roster_ids
        if pid in rec.protected | rec.unprojected
    ]
    return Recommendation(
        season=table.meta.season,
        week=table.meta.week,
        projections_version=table.meta.version,
        adds=[hits[a] for a in rec.adds],
        drops=[hits[d] for d in rec.drops],
        projected_gain=round(rec.projected_gain, 2),
        before=lineup_out(rec.before, hits),
        after=lineup_out(rec.after, hits),
        alternatives=[
            SwapOut(add=hits[s.add], drop=hits[s.drop], projected_gain=round(s.projected_gain, 2))
            for s in alternatives
        ],
        protected=protected,
        caveats=caveats(table.meta, len(free_agents), protected),
        evidence=EVIDENCE.get(request.max_moves),
    )
