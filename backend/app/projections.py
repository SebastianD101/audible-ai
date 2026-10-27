"""Read the weekly projections written offline by ``ml.serve.project``.

The API never trains anything: it serves the newest ``season=YYYY/week=WW.parquet`` (or a
pinned one) and its manifest. Loaded tables are cached by file modification time, so a new
weekly file is picked up on the next request without a restart.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pydantic import BaseModel

DEFAULT_PROJECTIONS_DIR = Path("ml/data/projections")
_FILE = re.compile(r"season=(\d{4})[\\/]week=(\d{2})\.parquet$")


class NoProjectionsError(RuntimeError):
    """No projections file exists yet."""


class ProjectionsMeta(BaseModel):
    """Which projections are being served, and where they came from."""

    season: int
    week: int
    version: str
    generated_at_utc: str
    rows: int
    model: str


class ProjectedPlayer(BaseModel):
    """One player's projection for the served week."""

    player_id: str
    name: str
    position: str
    team: str
    opponent: str
    projection: float
    hold_value: float


@dataclass(frozen=True)
class ProjectionTable:
    """A loaded projections file, keyed by player ID."""

    meta: ProjectionsMeta
    players: dict[str, ProjectedPlayer]

    def projections(self) -> dict[str, float]:
        """Player ID -> projected points, for the optimizer."""
        return {pid: p.projection for pid, p in self.players.items()}

    def hold_values(self) -> dict[str, float]:
        """Player ID -> season-to-date average, for the optimizer's drop guard."""
        return {pid: p.hold_value for pid, p in self.players.items()}


def latest_file(projections_dir: Path) -> Path:
    """The projections file with the highest (season, week).

    Raises:
        NoProjectionsError: with the command that creates one.
    """
    found = [
        (int(m[1]), int(m[2]), p)
        for p in projections_dir.glob("season=*/week=*.parquet")
        if (m := _FILE.search(p.as_posix()))
    ]
    if not found:
        raise NoProjectionsError(
            f"no projections in {projections_dir}; run "
            "`uv run python -m ml.serve.project --season <season> --week <week>`"
        )
    return max(found)[2]


def read_table(path: Path) -> ProjectionTable:
    """Read one projections file and its manifest."""
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    meta = ProjectionsMeta.model_validate(manifest)
    rows = pl.read_parquet(path).select(list(ProjectedPlayer.model_fields))
    players = {
        r["player_id"]: ProjectedPlayer.model_validate(r) for r in rows.iter_rows(named=True)
    }
    return ProjectionTable(meta=meta, players=players)


_cache: dict[Path, tuple[float, ProjectionTable]] = {}


def load_projections(projections_dir: Path) -> ProjectionTable:
    """The newest projections, re-read only when the file changes.

    Raises:
        NoProjectionsError: if there are none.
    """
    path = latest_file(projections_dir)
    mtime = path.stat().st_mtime
    cached = _cache.get(path)
    if cached is None or cached[0] != mtime:
        _cache[path] = (mtime, read_table(path))
    return _cache[path][1]
