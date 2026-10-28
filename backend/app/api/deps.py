"""Shared FastAPI dependencies: settings, projections, the player index, the API client."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import anthropic
from fastapi import Depends

from backend.app.config import Settings, get_settings
from backend.app.parsing.players import PlayerIndex, load_player_index
from backend.app.parsing.service import make_client
from backend.app.projections import ProjectionTable, load_projections
from backend.app.sleeper.client import SleeperClient

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_projections(settings: SettingsDep) -> ProjectionTable:
    """The projections being served (raises ``NoProjectionsError`` if none exist)."""
    return load_projections(settings.projections_dir)


ProjectionsDep = Annotated[ProjectionTable, Depends(get_projections)]


@lru_cache(maxsize=2)
def cached_index(season: int, rosters_dir: Path) -> PlayerIndex:
    """Build each season's player index once."""
    return load_player_index(season, rosters_dir)


def get_player_index(settings: SettingsDep, table: ProjectionsDep) -> PlayerIndex:
    """The player index for the served season."""
    return cached_index(table.meta.season, settings.rosters_dir)


PlayerIndexDep = Annotated[PlayerIndex, Depends(get_player_index)]


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic | None:
    """One API client per process (it pools connections); None when parsing with OCR."""
    return make_client(get_settings())


ClientDep = Annotated[anthropic.Anthropic | None, Depends(get_client)]


@lru_cache(maxsize=1)
def get_sleeper() -> SleeperClient:
    """One Sleeper client per process, so its response cache is shared across requests."""
    return SleeperClient(get_settings().sleeper_api_url)


SleeperDep = Annotated[SleeperClient, Depends(get_sleeper)]
