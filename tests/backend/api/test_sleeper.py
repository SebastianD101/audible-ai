from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.api.deps import get_sleeper
from backend.app.config import Settings
from backend.app.main import app
from tests.backend.api.test_parse import records
from tests.backend.api.world import SEASON
from tests.backend.sleeper.fake import LEAGUE_ID, USER_ID, league, routes
from tests.backend.sleeper.fake import client as fake_sleeper

Install = Callable[..., None]


@pytest.fixture
def use_sleeper(client: TestClient) -> Install:  # after `client`, which resets overrides
    """Serve the fake Sleeper API, optionally with some routes replaced."""

    def install(**changes: Any) -> None:
        fake = fake_sleeper(routes() | changes)
        app.dependency_overrides[get_sleeper] = lambda: fake

    return install


def test_leagues_lists_the_projected_season(client: TestClient, use_sleeper: Install) -> None:
    use_sleeper()
    r = client.get("/api/sleeper/leagues", params={"username": "someone"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["user_id"], body["season"]) == (USER_ID, SEASON)
    assert body["leagues"] == [
        {
            "league_id": LEAGUE_ID,
            "name": "Test League",
            "status": "in_season",
            "total_rosters": 2,
            "scoring": "PPR",
        }
    ]


def test_unknown_user_is_404(client: TestClient, use_sleeper: Install) -> None:
    use_sleeper()
    r = client.get("/api/sleeper/leagues", params={"username": "nobody"})
    assert r.status_code == 404
    assert "nobody" in r.json()["detail"]


def test_bad_username_never_reaches_sleeper(client: TestClient, use_sleeper: Install) -> None:
    use_sleeper()
    assert client.get("/api/sleeper/leagues", params={"username": "a/b"}).status_code == 422


def test_import_returns_review_rows_and_settings_and_stores_both(
    client: TestClient, settings: Settings, use_sleeper: Install
) -> None:
    use_sleeper()
    r = client.post("/api/sleeper/import", json={"league_id": LEAGUE_ID, "user_id": USER_ID})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["league_name"] == "Test League"
    roster = body["roster"]
    assert roster["kind"] == "roster"
    assert roster["roster"]["players"][0]["match"]["player_id"] == "qb1"
    assert roster["players"]["qb1"]["projection"] == 24.0
    assert body["free_agents"]["kind"] == "free_agents"
    assert body["settings"]["slots"]["WRRB_FLEX"] == 1
    assert body["warnings"] == []
    assert sorted(rec.kind for rec in records(settings)) == ["free_agents", "roster"]
    assert {rec.model for rec in records(settings)} == {"sleeper-api"}


def test_import_warns_about_scoring(client: TestClient, use_sleeper: Install) -> None:
    use_sleeper(**{f"/league/{LEAGUE_ID}": league(scoring_settings={"rec": 0.5})})
    r = client.post("/api/sleeper/import", json={"league_id": LEAGUE_ID, "user_id": USER_ID})
    assert "overvalued" in r.json()["warnings"][0]


def test_import_of_another_season_is_422(client: TestClient, use_sleeper: Install) -> None:
    use_sleeper(**{f"/league/{LEAGUE_ID}": league(season="2025")})
    r = client.post("/api/sleeper/import", json={"league_id": LEAGUE_ID, "user_id": USER_ID})
    assert r.status_code == 422
    assert "2025" in r.json()["detail"]


def test_import_for_someone_without_a_team_is_404(client: TestClient, use_sleeper: Install) -> None:
    use_sleeper()
    r = client.post("/api/sleeper/import", json={"league_id": LEAGUE_ID, "user_id": "333"})
    assert r.status_code == 404


def test_sleeper_down_is_503_with_retry_after(client: TestClient, use_sleeper: Install) -> None:
    use_sleeper(**{f"/league/{LEAGUE_ID}": httpx.ConnectError("down")})
    r = client.post("/api/sleeper/import", json={"league_id": LEAGUE_ID, "user_id": USER_ID})
    assert r.status_code == 503
    assert r.headers["retry-after"] == "30"


@pytest.mark.parametrize("league_id", ["abc", "1/2", ""])
def test_non_numeric_ids_are_422(client: TestClient, use_sleeper: Install, league_id: str) -> None:
    use_sleeper()
    r = client.post("/api/sleeper/import", json={"league_id": league_id, "user_id": USER_ID})
    assert r.status_code == 422
