"""Turn a Sleeper league into the same shapes a parsed screenshot produces.

- **Players** map to nflverse IDs through the ``sleeper_id`` crosswalk in ``rosters_weekly``, and
  team defenses map from Sleeper's team codes. These are exact IDs, so rows are ``clear`` and
  ``matched``. A player the crosswalk lacks becomes an unmatched row for the user to resolve.
- **Free agents** are every projected QB/RB/WR/TE not on any roster in the league, cut to the
  top ``FREE_AGENTS_PER_POSITION`` by projection. The cut loses nothing for up to three moves:
  a pickup's value depends only on its projection and position, so the best adds at a position
  are always its top-projected free agents.
- **Lineup slots** come from ``roster_positions``. Scoring the model can't represent (not full
  PPR, TE premium, 6-point passing TDs) becomes a warning that says which way it leans.
"""

from collections import Counter
from dataclasses import dataclass

from backend.app.parsing.players import IndexedPlayer, PlayerIndex
from backend.app.parsing.schemas import (
    Candidate,
    ExtractedPlayer,
    ParsedPlayer,
    ParsedRoster,
    PlayerMatch,
)
from backend.app.parsing.teams import canonical_team
from backend.app.projections import ProjectionTable
from backend.app.sleeper.schemas import SleeperLeague, SleeperRoster
from ml.optimize.league import LeagueSettings, SlotName

SLEEPER_MODEL = "sleeper-api"
FREE_AGENTS_PER_POSITION = 10
PROJECTED_POSITIONS = ("QB", "RB", "WR", "TE")
EMPTY_SLOT = "0"

# Sleeper roster position -> optimizer slot. K and DEF aren't projected, so they're left out.
SLOT_MAP: dict[str, SlotName] = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "FLEX": "FLEX",
    "WRRB_FLEX": "WRRB_FLEX", "SUPER_FLEX": "SUPERFLEX",
}  # fmt: skip
# Sleeper roster position -> the slot label shown in the review table.
SLOT_LABELS = {"SUPER_FLEX": "SUPERFLEX", "WRRB_FLEX": "W/R", "REC_FLEX": "W/T"}
NOT_STARTERS = {"BN", "IR", "TAXI"}
UNMODELED_STARTERS = {"K", "DEF"}  # expected and harmless: swaps never touch them


class NotInLeagueError(LookupError):
    """The user has no team in this league."""


@dataclass(frozen=True)
class LeagueImport:
    """A Sleeper league converted for review and recommendations."""

    roster: ParsedRoster
    free_agents: ParsedRoster
    settings: LeagueSettings
    warnings: list[str]


def resolve(sleeper_id: str, index: PlayerIndex) -> IndexedPlayer | None:
    """The index entry for a Sleeper player ID; team codes ("NYJ") are defenses."""
    if sleeper_id.isalpha():
        team = canonical_team(sleeper_id)
        return next((p for p in index.by_position("DST") if p.team == team), None)
    return index.by_sleeper_id.get(sleeper_id)


def parsed_row(player: IndexedPlayer | None, sleeper_id: str, slot: str | None) -> ParsedPlayer:
    """One review row: an exact match, or an unmatched row naming the Sleeper ID."""
    if player is None:
        return ParsedPlayer(
            extracted=ExtractedPlayer(
                name=f"Sleeper player {sleeper_id}",
                position=None,
                nfl_team=None,
                slot=slot,
                injury_tag=None,
                legibility="clear",
                note="Not in the nflverse ID crosswalk; pick the player or remove the row.",
            ),
            match=PlayerMatch(status="unmatched"),
            needs_review=True,
        )
    candidate = Candidate(
        player_id=player.player_id,
        name=player.name,
        position=player.position,
        team=player.team,
        score=1.0,
    )
    return ParsedPlayer(
        extracted=ExtractedPlayer(
            name=player.name,
            position=player.position,
            nfl_team=player.team,
            slot=slot,
            injury_tag=None,
            legibility="clear",
            note=None,
        ),
        match=PlayerMatch(
            status="matched", player_id=player.player_id, score=1.0, candidates=[candidate]
        ),
        needs_review=False,
    )


def find_roster(rosters: list[SleeperRoster], user_id: str) -> SleeperRoster:
    """The team this user owns or co-owns.

    Raises:
        NotInLeagueError: if they have no team here.
    """
    for roster in rosters:
        if roster.owner_id == user_id or user_id in (roster.co_owners or []):
            return roster
    raise NotInLeagueError("You don't have a team in this league.")


def roster_rows(roster: SleeperRoster, league: SleeperLeague, index: PlayerIndex) -> ParsedRoster:
    """The user's roster in Sleeper's order: starters by slot, then bench, IR, and taxi."""
    starter_slots = [p for p in league.roster_positions if p not in NOT_STARTERS]
    starters = roster.starters or []
    reserve, taxi = roster.reserve or [], roster.taxi or []
    placed = set(starters) | set(reserve) | set(taxi)
    bench = [p for p in roster.players or [] if p not in placed]

    order = [
        (sid, SLOT_LABELS.get(slot, slot))
        for sid, slot in zip(starters, starter_slots, strict=False)
        if sid != EMPTY_SLOT
    ]
    order += [(sid, "BN") for sid in bench] + [(sid, "IR") for sid in reserve]
    order += [(sid, "TAXI") for sid in taxi]
    players = [parsed_row(resolve(sid, index), sid, slot) for sid, slot in order]
    warnings = []
    if any(p.match.status == "unmatched" for p in players):
        warnings.append("Some Sleeper players aren't in the ID crosswalk; check the flagged rows.")
    return ParsedRoster(platform="sleeper", players=players, warnings=warnings, model=SLEEPER_MODEL)


def free_agent_rows(
    rosters: list[SleeperRoster],
    table: ProjectionTable,
    index: PlayerIndex,
    per_position: int = FREE_AGENTS_PER_POSITION,
) -> ParsedRoster:
    """The best-projected players at each position that nobody in the league rosters."""
    rostered_sleeper = {sid for r in rosters for sid in r.players or []}
    rostered = {p.player_id for sid in rostered_sleeper if (p := resolve(sid, index))}
    unknown = sum(resolve(sid, index) is None for sid in rostered_sleeper)
    indexed = {p.player_id: p for p in index.players}

    players = []
    for position in PROJECTED_POSITIONS:
        pool = sorted(
            (
                p
                for p in table.players.values()
                if p.position == position and p.player_id not in rostered and p.player_id in indexed
            ),
            key=lambda p: p.projection,
            reverse=True,
        )
        players += [parsed_row(indexed[p.player_id], "", None) for p in pool[:per_position]]
    warnings = []
    if unknown:
        warnings.append(
            f"{unknown} rostered player(s) in this league couldn't be identified, so one of "
            "these free agents might already be taken."
        )
    return ParsedRoster(platform="sleeper", players=players, warnings=warnings, model=SLEEPER_MODEL)


def league_settings(roster_positions: list[str]) -> tuple[LeagueSettings, list[str]]:
    """Optimizer slots from Sleeper's roster positions, with warnings for anything approximated.

    Raises:
        ValueError: if the league has no QB/RB/WR/TE starting slots at all.
    """
    counts = Counter(roster_positions)
    slots: Counter[SlotName] = Counter()
    for position, n in counts.items():
        if position in SLOT_MAP:
            slots[SLOT_MAP[position]] += n
    warnings = []
    flex_positions = frozenset({"RB", "WR", "TE"})
    if counts["REC_FLEX"]:
        slots["FLEX"] += counts["REC_FLEX"]
        if counts["FLEX"] or counts["WRRB_FLEX"]:
            warnings.append(
                "The W/T flex is treated as a regular FLEX (RB/WR/TE) alongside your other flex "
                "slots, so a running back could be suggested there."
            )
        else:
            flex_positions = frozenset({"WR", "TE"})
    other = sorted(set(counts) - set(SLOT_MAP) - {"REC_FLEX"} - NOT_STARTERS - UNMODELED_STARTERS)
    if other:
        warnings.append(f"Slots {', '.join(other)} aren't projected and are ignored.")
    settings = LeagueSettings(slots=dict(slots), flex_positions=flex_positions)
    return settings, warnings


def scoring_summary(scoring: dict[str, float]) -> str:
    """A short label like "PPR", "Half PPR", or "0.25 PPR"."""
    rec = scoring.get("rec", 0.0)
    names = {1.0: "PPR", 0.5: "Half PPR", 0.0: "Standard"}
    return names.get(rec, f"{rec:g} PPR")


def scoring_warnings(scoring: dict[str, float]) -> list[str]:
    """Where the league's scoring differs from the full-PPR, 4-point-passing-TD projections."""
    warnings = []
    rec = scoring.get("rec", 0.0)
    if rec != 1.0:
        lean = "overvalued" if rec < 1.0 else "undervalued"
        warnings.append(
            f"Your league scores {rec:g} per catch, but projections are full PPR, so pass "
            f"catchers are {lean} relative to your scoring."
        )
    te_bonus = scoring.get("bonus_rec_te", 0.0)
    if te_bonus:
        warnings.append(
            f"Your league's TE premium (+{te_bonus:g} per catch) isn't modeled, so tight ends "
            "are undervalued."
        )
    pass_td = scoring.get("pass_td", 4.0)
    if pass_td != 4.0:
        lean = "undervalued" if pass_td > 4.0 else "overvalued"
        warnings.append(
            f"Your league scores {pass_td:g} per passing TD, but projections use 4, so "
            f"quarterbacks are {lean}."
        )
    return warnings


def import_league(
    league: SleeperLeague,
    rosters: list[SleeperRoster],
    user_id: str,
    index: PlayerIndex,
    table: ProjectionTable,
) -> LeagueImport:
    """Convert one league for this user.

    Raises:
        NotInLeagueError: the user has no team in the league.
        ValueError: the league has no projected starting slots.
    """
    roster = roster_rows(find_roster(rosters, user_id), league, index)
    settings, slot_warnings = league_settings(league.roster_positions)
    return LeagueImport(
        roster=roster,
        free_agents=free_agent_rows(rosters, table, index),
        settings=settings,
        warnings=scoring_warnings(league.scoring_settings) + slot_warnings,
    )
