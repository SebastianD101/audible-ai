"""Pydantic models for an imported roster, as shown on the review screen.

``ExtractedPlayer`` is a row as the source (a Sleeper league) describes it; ``ParsedRoster``
adds the match to nflverse IDs and a ``needs_review`` flag per player. It is the contract the
review screen and the optimizer consume. The names predate Sleeper import, when rows were read
from screenshots, and are kept so the API shape is unchanged.
"""

from typing import Literal

from pydantic import BaseModel, Field

Position = Literal["QB", "RB", "WR", "TE", "K", "DST"]
Platform = Literal["espn", "yahoo", "sleeper", "other", "unknown"]
Legibility = Literal["clear", "partial", "guessed"]
MatchStatus = Literal["matched", "ambiguous", "unmatched"]


class ExtractedPlayer(BaseModel):
    """One roster row as the source describes it, before any ID matching."""

    name: str = Field(description="Player name exactly as shown, or the team name for a defense.")
    position: Position | None = Field(
        description="Fantasy position shown for the player; null if not visible."
    )
    nfl_team: str | None = Field(
        description="NFL team abbreviation shown next to the player (e.g. 'KC'); null if not shown."
    )
    slot: str | None = Field(
        description="Lineup slot as shown (e.g. 'QB', 'RB', 'FLEX', 'BN', 'IR'); null if not shown."
    )
    injury_tag: str | None = Field(
        description="Injury designation shown (e.g. 'Q', 'O', 'IR'); null if none is shown."
    )
    legibility: Legibility = Field(
        description=(
            "'clear' if every field was read without doubt, 'partial' if some text was cut off "
            "or blurry, 'guessed' if the name itself is uncertain."
        )
    )
    note: str | None = Field(description="Short reason for any uncertainty; null if clear.")


class Candidate(BaseModel):
    """A possible nflverse player for an extracted name."""

    player_id: str
    name: str
    position: Position
    team: str | None
    score: float = Field(ge=0.0, le=1.0)


class PlayerMatch(BaseModel):
    """Result of matching one extracted player to the player index."""

    status: MatchStatus
    player_id: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    candidates: list[Candidate] = Field(default_factory=list)


class ParsedPlayer(BaseModel):
    """An extracted player with its match and whether the user should check it."""

    extracted: ExtractedPlayer
    match: PlayerMatch
    needs_review: bool


class ParsedRoster(BaseModel):
    """Validated, ID-matched roster returned to the client for review."""

    platform: Platform
    players: list[ParsedPlayer]
    warnings: list[str]
    model: str
