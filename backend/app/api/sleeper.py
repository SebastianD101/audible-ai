"""Import a roster, free agents, and lineup settings from a Sleeper league.

This is the exact alternative to screenshot parsing for Sleeper leagues: player IDs come from
the nflverse crosswalk instead of OCR, and the free-agent pool is the whole league's unrostered
players instead of what fit on screen. The result reuses the parse response shape, so the review
table works unchanged.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.api.deps import PlayerIndexDep, ProjectionsDep, SleeperDep
from backend.app.api.parse import RETRY_AFTER_SECONDS, ParseResponse, store_parse
from backend.app.db.session import get_session
from backend.app.sleeper.client import SleeperNotFoundError, SleeperUnavailableError
from backend.app.sleeper.importer import NotInLeagueError, import_league, scoring_summary
from ml.optimize.league import LeagueSettings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sleeper", tags=["sleeper"])

SLEEPER_ID = r"^\d{1,32}$"


class SleeperLeagueOut(BaseModel):
    """One of the user's leagues, for the league picker."""

    league_id: str
    name: str
    status: str
    total_rosters: int
    scoring: str


class SleeperLeaguesResponse(BaseModel):
    """A Sleeper user and their leagues for the season being projected."""

    user_id: str
    username: str
    display_name: str
    season: int
    leagues: list[SleeperLeagueOut]


class SleeperImportRequest(BaseModel):
    """Which league to import, for which user."""

    league_id: str = Field(pattern=SLEEPER_ID)
    user_id: str = Field(pattern=SLEEPER_ID)


class SleeperImportResponse(BaseModel):
    """The user's roster and the league's best free agents, ready for review, plus settings."""

    league_name: str
    roster: ParseResponse
    free_agents: ParseResponse
    settings: LeagueSettings
    warnings: list[str]


def sleeper_error(e: SleeperNotFoundError | SleeperUnavailableError) -> HTTPException:
    """404 for an unknown user or league, 503 (retry later) when Sleeper is down."""
    if isinstance(e, SleeperNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        str(e),
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )


@router.get("/leagues")
def list_leagues(
    sleeper: SleeperDep,
    table: ProjectionsDep,
    username: Annotated[str, Query(min_length=1, max_length=40, pattern=r"^[\w.-]+$")],
) -> SleeperLeaguesResponse:
    """The user's leagues in the season the served projections cover."""
    season = table.meta.season
    try:
        user = sleeper.user(username)
        leagues = sleeper.leagues(user.user_id, season)
    except (SleeperNotFoundError, SleeperUnavailableError) as e:
        raise sleeper_error(e) from e
    return SleeperLeaguesResponse(
        user_id=user.user_id,
        username=user.username or username,
        display_name=user.display_name or user.username or username,
        season=season,
        leagues=[
            SleeperLeagueOut(
                league_id=lg.league_id,
                name=lg.name,
                status=lg.status,
                total_rosters=lg.total_rosters,
                scoring=scoring_summary(lg.scoring_settings),
            )
            for lg in leagues
        ],
    )


@router.post("/import")
def import_sleeper_league(
    request: SleeperImportRequest,
    sleeper: SleeperDep,
    table: ProjectionsDep,
    index: PlayerIndexDep,
    session: Annotated[Session, Depends(get_session)],
) -> SleeperImportResponse:
    """Import the user's team and the league's free agents; both are stored like parses."""
    try:
        league = sleeper.league(request.league_id)
        rosters = sleeper.rosters(request.league_id)
    except (SleeperNotFoundError, SleeperUnavailableError) as e:
        raise sleeper_error(e) from e
    if league.season != table.meta.season:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"This is a {league.season} league; projections are for {table.meta.season}.",
        )
    try:
        imported = import_league(league, rosters, request.user_id, index, table)
    except NotInLeagueError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    except ValueError as e:  # no projected starting slots
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    logger.info("imported Sleeper league %s", league.league_id)
    return SleeperImportResponse(
        league_name=league.name,
        roster=store_parse(session, "roster", imported.roster, index, table),
        free_agents=store_parse(session, "free_agents", imported.free_agents, index, table),
        settings=imported.settings,
        warnings=imported.warnings,
    )
