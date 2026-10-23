"""The single information-cutoff rule every feature obeys.

Features for target week W are built as of the Tuesday night after week W-1, when waiver claims
process. At that point every game and report for weeks before W is known, and nothing for week W
is: no week-W closing lines, injury designations, or practice reports. So the whole rule is
"rows with (season, week) < (W.season, W.week)", applied by :func:`available_before` at the top
of every feature function.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, order=True)
class Week:
    """A target week: features may only use data from strictly earlier weeks."""

    season: int
    week: int

    def __post_init__(self) -> None:
        """Reject impossible week numbers."""
        if self.week < 1:
            raise ValueError(f"week must be >= 1, got {self.week}")

    def __str__(self) -> str:
        """Format as ``2023-W10``."""
        return f"{self.season}-W{self.week:02d}"


def before_expr(as_of: Week) -> pl.Expr:
    """Return a predicate selecting rows from weeks strictly before ``as_of``."""
    return (pl.col("season") < as_of.season) | (
        (pl.col("season") == as_of.season) & (pl.col("week") < as_of.week)
    )


def available_before[Frame: (pl.DataFrame, pl.LazyFrame)](df: Frame, as_of: Week) -> Frame:
    """Keep only rows from weeks strictly before ``as_of`` (the waiver-time cutoff)."""
    return df.filter(before_expr(as_of))
