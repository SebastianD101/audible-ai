"""The parts of Sleeper's API responses that the importer reads (other fields are ignored)."""

from pydantic import BaseModel, ConfigDict, Field


class _Sleeper(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SleeperUser(_Sleeper):
    """A Sleeper account."""

    user_id: str
    username: str | None = None
    display_name: str | None = None


class SleeperLeague(_Sleeper):
    """One league's identity, lineup slots, and scoring."""

    league_id: str
    name: str
    season: int
    status: str
    total_rosters: int
    roster_positions: list[str] = Field(default_factory=list)
    scoring_settings: dict[str, float] = Field(default_factory=dict)


class SleeperRoster(_Sleeper):
    """One team in a league. Sleeper sends null for empty lists; "0" marks an empty starter."""

    roster_id: int
    owner_id: str | None = None
    co_owners: list[str] | None = None
    starters: list[str] | None = None
    players: list[str] | None = None
    reserve: list[str] | None = None
    taxi: list[str] | None = None
