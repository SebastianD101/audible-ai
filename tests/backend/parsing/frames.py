"""Small in-memory weekly-roster frames for parsing tests."""

import polars as pl

from backend.app.parsing.schemas import ExtractedPlayer

ROWS: list[tuple[str, int, str, str, str]] = [
    # gsis_id, week, full_name, position, team
    ("00-001", 1, "Justin Jefferson", "WR", "MIN"),
    ("00-002", 1, "Marvin Harrison Jr.", "WR", "ARI"),
    ("00-003", 1, "A.J. Brown", "WR", "PHI"),
    ("00-004", 1, "Josh Allen", "QB", "BUF"),
    ("00-005", 1, "Josh Allen", "LB", "JAX"),  # same name, non-fantasy position
    ("00-006", 1, "Mike Williams", "WR", "NYJ"),
    ("00-007", 1, "Mike Williams", "WR", "PIT"),
    ("00-008", 1, "Amon-Ra St. Brown", "WR", "DET"),
    ("00-009", 1, "Brandon Aubrey", "K", "DAL"),
    ("00-010", 1, "Kenneth Walker III", "RB", "SEA"),
    ("00-011", 1, "Davante Adams", "WR", "LV"),
    ("00-011", 5, "Davante Adams", "WR", "LA"),  # traded mid-season
    ("00-012", 1, "Sam LaPorta", "TE", "DET"),
    ("00-013", 1, "Justin Tucker", "K", "BAL"),
    ("00-014", 1, "Justin Herbert", "QB", "LAC"),
]


def rosters(rows: list[tuple[str, int, str, str, str]] | None = None) -> pl.DataFrame:
    """A ``rosters_weekly``-shaped frame for season 2026."""
    data = ROWS if rows is None else rows
    return pl.DataFrame(
        {
            "season": [2026] * len(data),
            "week": [r[1] for r in data],
            "gsis_id": [r[0] for r in data],
            "full_name": [r[2] for r in data],
            "position": [r[3] for r in data],
            "team": [r[4] for r in data],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )


def extracted(
    name: str,
    position: str | None = None,
    team: str | None = None,
    legibility: str = "clear",
) -> ExtractedPlayer:
    """An ``ExtractedPlayer`` with only the fields matching cares about."""
    return ExtractedPlayer.model_validate(
        {
            "name": name,
            "position": position,
            "nfl_team": team,
            "slot": None,
            "injury_tag": None,
            "legibility": legibility,
            "note": None,
        }
    )
