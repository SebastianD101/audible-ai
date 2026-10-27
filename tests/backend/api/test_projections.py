from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.projections import NoProjectionsError, latest_file, load_projections
from tests.backend.api.world import SEASON, WEEK


def test_meta_reports_the_served_week(client: TestClient) -> None:
    response = client.get("/api/projections/meta")
    assert response.status_code == 200
    body = response.json()
    assert (body["season"], body["week"], body["version"]) == (SEASON, WEEK, "abc123def456")


def test_meta_is_503_with_instructions_when_no_projections(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    settings.projections_dir = tmp_path / "empty"
    response = client.get("/api/projections/meta")
    assert response.status_code == 503
    assert "ml.serve.project" in response.json()["detail"]


def test_latest_file_picks_highest_season_and_week(tmp_path: Path) -> None:
    for season, week in [(2026, 7), (2026, 10), (2025, 17)]:
        path = tmp_path / f"season={season}" / f"week={week:02d}.parquet"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"")
    assert latest_file(tmp_path).as_posix().endswith("season=2026/week=10.parquet")


def test_load_projections_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(NoProjectionsError):
        load_projections(tmp_path)


def test_search_prefix_finds_player_with_projection(client: TestClient) -> None:
    hits = client.get("/api/players/search", params={"q": "jah"}).json()
    assert hits[0]["player_id"] == "rb1"
    assert hits[0]["projection"] == 21.0


def test_search_fuzzy_and_last_name(client: TestClient) -> None:
    hits = client.get("/api/players/search", params={"q": "Puka Nakua"}).json()
    assert hits[0]["name"] == "Puka Nacua"
    names = [h["name"] for h in client.get("/api/players/search", params={"q": "brown"}).json()]
    assert names[0] == "Amon-Ra St. Brown"


def test_search_includes_unprojected_players_with_null_projection(client: TestClient) -> None:
    hits = client.get("/api/players/search", params={"q": "aubrey"}).json()
    assert hits[0]["player_id"] == "k1"
    assert hits[0]["projection"] is None


def test_search_filters_by_position_and_validates_query(client: TestClient) -> None:
    hits = client.get("/api/players/search", params={"q": "j", "position": "QB"}).json()
    assert hits
    assert {h["position"] for h in hits} == {"QB"}
    assert client.get("/api/players/search", params={"q": ""}).status_code == 422
