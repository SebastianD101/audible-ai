"""In-memory index of matchable players, built from nflverse weekly rosters.

Each player appears once, at their latest rostered week in the season, so trades and
practice-squad moves resolve to the current team. Only fantasy positions are kept. Team
defenses are added as synthetic ``DST:<TEAM>`` entries because nflverse has no defense rows.
"""

import unicodedata
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import get_args

import polars as pl

from backend.app.parsing.schemas import Position
from backend.app.parsing.teams import TEAMS

DEFAULT_ROSTERS_DIR = Path("ml/data/raw/rosters_weekly")

PLAYER_POSITIONS: frozenset[str] = frozenset(get_args(Position)) - {"DST"}
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes: "Marvin Harrison Jr." -> "marvin harrison".

    Periods and apostrophes are removed without a space ("A.J." -> "aj", "Ma'ae" -> "maae") so
    platforms that drop them still match; hyphens become spaces.
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    cleaned = ascii_name.lower().replace(".", "").replace("'", "").replace("-", " ")
    cleaned = "".join(ch if ch.isalnum() or ch == " " else " " for ch in cleaned)
    tokens = cleaned.split()
    while len(tokens) > 1 and tokens[-1] in _SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


@dataclass(frozen=True)
class IndexedPlayer:
    """One matchable player (or team defense)."""

    player_id: str
    name: str
    position: Position
    team: str | None
    key: str  # normalize_name(name)
    aliases: tuple[str, ...] = ()  # other normalized names that should match exactly
    sleeper_id: str | None = None  # Sleeper's player ID, from the nflverse crosswalk


@dataclass(frozen=True)
class PlayerIndex:
    """All matchable players for one season."""

    season: int
    players: tuple[IndexedPlayer, ...]

    def by_position(self, position: Position | None) -> tuple[IndexedPlayer, ...]:
        """Players at ``position``, or every non-defense player when position is unknown."""
        if position is None:
            return tuple(p for p in self.players if p.position != "DST")
        return tuple(p for p in self.players if p.position == position)

    @cached_property
    def by_sleeper_id(self) -> dict[str, IndexedPlayer]:
        """Sleeper player ID -> player, for players the crosswalk covers."""
        return {p.sleeper_id: p for p in self.players if p.sleeper_id is not None}


def defense_entries() -> tuple[IndexedPlayer, ...]:
    """One synthetic entry per team defense, matchable by nickname, full name, or abbreviation."""
    return tuple(
        IndexedPlayer(
            player_id=f"DST:{abbr}",
            name=f"{location} {nickname}",
            position="DST",
            team=abbr,
            key=normalize_name(nickname),
            aliases=(normalize_name(f"{location} {nickname}"), abbr.lower()),
        )
        for abbr, (location, nickname) in TEAMS.items()
    )


def build_player_index(rosters: pl.DataFrame, season: int) -> PlayerIndex:
    """Build the index from one season of ``rosters_weekly`` rows.

    Rows without a ``gsis_id`` or outside fantasy positions are dropped; each ``gsis_id`` keeps
    its latest week, which also collapses upstream duplicate-ID rows to a single entry.
    """
    if "sleeper_id" not in rosters.columns:
        rosters = rosters.with_columns(pl.lit(None, dtype=pl.String).alias("sleeper_id"))
    latest = (
        rosters.filter(
            (pl.col("season") == season)
            & pl.col("gsis_id").is_not_null()
            & pl.col("full_name").is_not_null()
            & pl.col("position").is_in(sorted(PLAYER_POSITIONS))
        )
        .sort(["gsis_id", "week"])
        # Keep the last known Sleeper ID if the latest week's row lacks one.
        .with_columns(pl.col("sleeper_id").forward_fill().over("gsis_id"))
        .unique(subset="gsis_id", keep="last", maintain_order=True)
    )
    players = tuple(
        IndexedPlayer(
            player_id=row["gsis_id"],
            name=row["full_name"],
            position=row["position"],
            team=row["team"],
            key=normalize_name(row["full_name"]),
            sleeper_id=row["sleeper_id"],
        )
        for row in latest.select(
            "gsis_id", "full_name", "position", "team", "sleeper_id"
        ).iter_rows(named=True)
    )
    return PlayerIndex(season=season, players=players + defense_entries())


def load_player_index(season: int, rosters_dir: Path = DEFAULT_ROSTERS_DIR) -> PlayerIndex:
    """Read ``season=YYYY.parquet`` from the raw rosters layer and index it.

    Raises:
        FileNotFoundError: if that season has not been ingested yet.
    """
    path = rosters_dir / f"season={season}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `uv run python -m ml.ingest.rosters_weekly "
            f"--start-season {season} --end-season {season}`"
        )
    return build_player_index(pl.read_parquet(path), season)
