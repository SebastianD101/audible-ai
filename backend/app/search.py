"""Player search for correcting parsed rows and building the free-agent list.

Searches the same player index the parser matches against (every rostered player this season,
including kickers, defenses, and players on bye, who have no projection), so anything the user
can see in their league can be found. A typed prefix of any name token ranks first, so "jah"
finds Jahmyr Gibbs; otherwise the parser's fuzzy name score applies.
"""

from pydantic import BaseModel

from backend.app.parsing.match import name_score, ratio
from backend.app.parsing.players import IndexedPlayer, PlayerIndex, normalize_name
from backend.app.parsing.schemas import Position
from backend.app.projections import ProjectionTable

MIN_SCORE = 0.6
PREFIX_SCORE = 1.0


class PlayerHit(BaseModel):
    """A search result, with this week's projection when the player has one."""

    player_id: str
    name: str
    position: Position
    team: str | None
    projection: float | None
    hold_value: float | None


def search_score(query: str, player: IndexedPlayer) -> float:
    """Similarity in [0, 1] of a normalized query to a player."""
    keys = (player.key, *player.aliases)
    if any(k.startswith(query) or f" {query}" in f" {k}" for k in keys):
        return PREFIX_SCORE
    return max(name_score(query, player), max(ratio(query, k) for k in keys))


def to_hit(player: IndexedPlayer, table: ProjectionTable) -> PlayerHit:
    """Attach the served week's projection (None for players without one)."""
    projected = table.players.get(player.player_id)
    return PlayerHit(
        player_id=player.player_id,
        name=player.name,
        position=player.position,
        team=player.team,
        projection=projected.projection if projected else None,
        hold_value=projected.hold_value if projected else None,
    )


def search_players(
    index: PlayerIndex,
    table: ProjectionTable,
    query: str,
    position: Position | None = None,
    limit: int = 10,
) -> list[PlayerHit]:
    """Best matches for ``query``; ties go to the higher projection."""
    q = normalize_name(query)
    if not q:
        return []
    pool = index.players if position is None else index.by_position(position)
    scored = [(s, p) for p in pool if (s := search_score(q, p)) >= MIN_SCORE]

    def projection(p: IndexedPlayer) -> float:
        hit = table.players.get(p.player_id)
        return hit.projection if hit else -1.0

    scored.sort(key=lambda t: (-t[0], -projection(t[1]), t[1].name))
    return [to_hit(p, table) for _, p in scored[:limit]]


def hits_for(index: PlayerIndex, table: ProjectionTable, ids: set[str]) -> dict[str, PlayerHit]:
    """Display info and projections for the given player IDs (unknown IDs are skipped)."""
    return {p.player_id: to_hit(p, table) for p in index.players if p.player_id in ids}
