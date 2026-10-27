"""Projections metadata and player search."""

from typing import Annotated

from fastapi import APIRouter, Query

from backend.app.api.deps import PlayerIndexDep, ProjectionsDep
from backend.app.parsing.schemas import Position
from backend.app.projections import ProjectionsMeta
from backend.app.search import PlayerHit, search_players

router = APIRouter(prefix="/api", tags=["projections"])


@router.get("/projections/meta")
def projections_meta(table: ProjectionsDep) -> ProjectionsMeta:
    """Which season, week, and projections version is being served."""
    return table.meta


@router.get("/players/search")
def player_search(
    table: ProjectionsDep,
    index: PlayerIndexDep,
    q: Annotated[str, Query(min_length=1, max_length=64)],
    position: Position | None = None,
    limit: Annotated[int, Query(ge=1, le=25)] = 10,
) -> list[PlayerHit]:
    """Players whose names match ``q``, with this week's projection when they have one."""
    return search_players(index, table, q, position, limit)
