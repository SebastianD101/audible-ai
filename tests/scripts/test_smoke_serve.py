"""The deploy smoke test builds a realistic request from the newest committed projections."""

from pathlib import Path

import polars as pl
import pytest

from scripts.smoke_serve import FREE_AGENTS, ROSTER_SIZE, latest_projections, recommendation_request


def write_week(root: Path, week: int, n: int) -> None:
    path = root / "season=2026" / f"week={week:02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [f"w{week}p{i}" for i in range(n)]
    pl.DataFrame({"player_id": ids, "projection": [float(i) for i in range(n)]}).write_parquet(path)


def test_reads_the_newest_week(tmp_path: Path) -> None:
    write_week(tmp_path, 9, 3)
    write_week(tmp_path, 10, 3)
    assert latest_projections(tmp_path)["player_id"].to_list()[0].startswith("w10")


def test_no_projections_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        latest_projections(tmp_path)


def test_roster_is_the_top_tier_and_free_agents_the_next() -> None:
    n = ROSTER_SIZE + FREE_AGENTS + 10
    df = pl.DataFrame({"player_id": [f"p{i}" for i in range(n)], "projection": list(range(n))})
    body = recommendation_request(df)
    assert body["roster"][0] == f"p{n - 1}"
    assert len(body["roster"]) == ROSTER_SIZE
    assert len(body["free_agents"]) == FREE_AGENTS
    assert not set(body["roster"]) & set(body["free_agents"])
