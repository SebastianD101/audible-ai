"""League settings the optimizer needs: scoring and starting lineup slots.

Slots are data, so a 2-TE or superflex league is a settings change, not a code change. Only
positions the projection model covers (QB/RB/WR/TE) can have slots; K and DST are not
projected, so they are left out of lineup value entirely (a swap never changes them).
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ml.features.load import FANTASY_POSITIONS

SlotName = Literal["QB", "RB", "WR", "TE", "WRRB_FLEX", "FLEX", "SUPERFLEX"]

# Fill order for the greedy lineup: dedicated slots, then the RB/WR flex, FLEX, and SUPERFLEX.
# Each later slot type's eligible positions contain the earlier ones', which is what makes
# greedy optimal.
SLOT_ORDER: tuple[SlotName, ...] = ("QB", "RB", "WR", "TE", "WRRB_FLEX", "FLEX", "SUPERFLEX")
WRRB_POSITIONS = frozenset({"RB", "WR"})


def default_slots() -> dict[SlotName, int]:
    """A standard one-QB PPR lineup: QB, 2 RB, 2 WR, TE, FLEX."""
    return {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}


@dataclass(frozen=True)
class Player:
    """A player the optimizer can start, add, or drop."""

    player_id: str
    position: str


class LeagueSettings(BaseModel):
    """Scoring and lineup configuration for one league."""

    scoring: Literal["ppr"] = Field(
        default="ppr", description="Only PPR is supported: the projection model is PPR-only."
    )
    slots: dict[SlotName, int] = Field(default_factory=default_slots)
    flex_positions: frozenset[str] = frozenset({"RB", "WR", "TE"})

    @field_validator("slots", mode="before")
    @classmethod
    def reject_unprojected_slots(cls, value: object) -> object:
        """Explain, rather than just reject, slots for positions the model can't value."""
        if isinstance(value, dict):
            unsupported = sorted(set(value) & {"K", "DST", "DEF", "IDP"})
            if unsupported:
                raise ValueError(
                    f"slots {unsupported} are not projected; leave them out (swaps never "
                    "touch them, so recommendations are unaffected)"
                )
        return value

    @field_validator("slots")
    @classmethod
    def non_negative_counts(cls, value: dict[SlotName, int]) -> dict[SlotName, int]:
        """Slot counts must be non-negative and at least one slot must exist."""
        if any(n < 0 for n in value.values()) or sum(value.values()) == 0:
            raise ValueError("slot counts must be >= 0 with at least one starting slot")
        return value

    @model_validator(mode="after")
    def flex_positions_are_projected(self) -> "LeagueSettings":
        """FLEX may only draw on projected, non-QB positions."""
        if not self.flex_positions or not self.flex_positions <= set(FANTASY_POSITIONS) - {"QB"}:
            raise ValueError("flex_positions must be a non-empty subset of RB/WR/TE")
        if self.slots.get("WRRB_FLEX", 0) > 0 and not self.flex_positions >= WRRB_POSITIONS:
            # Otherwise WRRB_FLEX and FLEX aren't nested and greedy is no longer exact.
            raise ValueError("a WRRB_FLEX slot needs flex_positions to include RB and WR")
        return self

    def eligible(self, slot: SlotName) -> frozenset[str]:
        """Positions that may fill ``slot``."""
        if slot == "WRRB_FLEX":
            return WRRB_POSITIONS
        if slot == "FLEX":
            return self.flex_positions
        if slot == "SUPERFLEX":
            return frozenset(FANTASY_POSITIONS)
        return frozenset({slot})

    def ordered_slots(self) -> list[tuple[SlotName, int]]:
        """Configured slots with a positive count, in greedy fill order."""
        return [(s, self.slots[s]) for s in SLOT_ORDER if self.slots.get(s, 0) > 0]
