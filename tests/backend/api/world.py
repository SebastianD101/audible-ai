"""A small served week: rosters for the player index and a matching projections file."""

import json
from pathlib import Path

import polars as pl

SEASON, WEEK = 2026, 8

# player_id, name, position, team, projection (None: rostered but not projected), hold_value
PLAYERS: list[tuple[str, str, str, str, float | None, float]] = [
    ("qb1", "Josh Allen", "QB", "BUF", 24.0, 26.0),
    ("qb2", "Drake Maye", "QB", "NE", 21.0, 19.0),
    ("qb3", "Justin Fields", "QB", "KC", 1.0, 15.0),  # hurt this week
    ("rb1", "Jahmyr Gibbs", "RB", "DET", 21.0, 28.0),
    ("rb2", "Kyren Williams", "RB", "LA", 18.0, 19.0),
    ("rb3", "James Cook", "RB", "BUF", 17.0, 18.0),
    ("rb4", "Kyle Monangai", "RB", "CHI", 3.0, 14.0),
    ("rb5", "Bench Back", "RB", "NYG", 4.0, 4.0),
    ("wr1", "Puka Nacua", "WR", "LA", 22.0, 22.0),
    ("wr2", "Zay Flowers", "WR", "BAL", 19.0, 20.0),
    ("wr3", "Amon-Ra St. Brown", "WR", "DET", 19.5, 21.0),
    ("wr4", "Depth Receiver", "WR", "TEN", 3.0, 3.0),
    ("wr5", "Waiver Wideout", "WR", "MIA", 14.0, 9.0),
    ("wr6", "Bye Week Guy", "WR", "SEA", None, 12.0),
    ("te1", "Trey McBride", "TE", "ARI", 18.0, 18.0),
    ("te2", "Backup Tightend", "TE", "CLE", 5.0, 5.0),
    ("te3", "Streaming Tightend", "TE", "GB", 11.0, 10.0),
    ("k1", "Brandon Aubrey", "K", "DAL", None, 0.0),
]


def sleeper_id(player_id: str) -> str:
    """A made-up numeric Sleeper ID per player: its index in ``PLAYERS`` plus 1000."""
    return str(1000 + [p[0] for p in PLAYERS].index(player_id))


def write_world(root: Path) -> tuple[Path, Path]:
    """Write ``rosters_weekly`` and one projections file; return (rosters_dir, projections_dir)."""
    rosters_dir, projections_dir = root / "rosters", root / "projections"
    rosters_dir.mkdir()
    pl.DataFrame(
        {
            "season": [SEASON] * len(PLAYERS),
            "week": [WEEK] * len(PLAYERS),
            "gsis_id": [p[0] for p in PLAYERS],
            "full_name": [p[1] for p in PLAYERS],
            "position": [p[2] for p in PLAYERS],
            "team": [p[3] for p in PLAYERS],
            "sleeper_id": [sleeper_id(p[0]) for p in PLAYERS],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    ).write_parquet(rosters_dir / f"season={SEASON}.parquet")

    projected = [p for p in PLAYERS if p[4] is not None]
    week_dir = projections_dir / f"season={SEASON}"
    week_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "player_id": [p[0] for p in projected],
            "name": [p[1] for p in projected],
            "position": [p[2] for p in projected],
            "season": [SEASON] * len(projected),
            "week": [WEEK] * len(projected),
            "team": [p[3] for p in projected],
            "opponent": ["OPP"] * len(projected),
            "projection": [p[4] for p in projected],
            "hold_value": [p[5] for p in projected],
        }
    ).write_parquet(week_dir / f"week={WEEK:02d}.parquet")
    manifest = {
        "season": SEASON,
        "week": WEEK,
        "version": "abc123def456",
        "sha256": "abc123def456" + "0" * 52,
        "rows": len(projected),
        "generated_at_utc": "2026-10-27T23:00:00+00:00",
        "model": "lgbm",
    }
    (week_dir / f"week={WEEK:02d}.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rosters_dir, projections_dir
