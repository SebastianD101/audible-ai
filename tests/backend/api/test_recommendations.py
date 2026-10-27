from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.db.models import RecommendationRecord

# qb3 and rb4 are hurt this week (projection 1 and 3) but have season-long value; wr6 is on bye
# (no projection); k1 is a kicker. See tests/backend/api/world.py.
ROSTER = ["qb1", "qb3", "rb2", "rb4", "rb5", "wr2", "wr3", "wr4", "wr6", "te2", "k1"]
FREE_AGENTS = ["wr5", "te3", "qb2", "rb3"]


def recommend(client: TestClient, **overrides: Any) -> Any:
    body = {"roster": ROSTER, "free_agents": FREE_AGENTS} | overrides
    return client.post("/api/recommendations", json=body)


def by_id(players: list[dict[str, Any]]) -> list[str]:
    return [p["player_id"] for p in players]


def test_recommends_best_add_and_drops_the_least_valuable_bench_player(
    client: TestClient,
) -> None:
    r = recommend(client)
    assert r.status_code == 200, r.text
    body = r.json()
    # rb3 (17) starts at RB and pushes rb5 (4) to FLEX over rb4 (3): +13 + 1.
    assert by_id(body["adds"]) == ["rb3"]
    assert body["projected_gain"] == pytest.approx(14.0)
    # wr4 (3, season value 3) goes, not qb3 (1 this week, but 15 per game this season).
    assert by_id(body["drops"]) == ["wr4"]
    assert body["after"]["points"] - body["before"]["points"] == pytest.approx(14.0)
    assert body["alternatives"][0]["add"]["player_id"] == "rb3"
    assert body["evidence"]["realized_gain"] == pytest.approx(1.98)
    assert (body["season"], body["week"]) == (2026, 8)


def test_explains_protected_players(client: TestClient) -> None:
    protected = {
        p["player"]["player_id"]: p["reason"] for p in recommend(client).json()["protected"]
    }
    assert protected["rb4"] == "season_starter"  # hurt now, but starts on season value
    assert protected["wr6"] == "no_projection"  # bye
    assert protected["k1"] == "no_projection"
    assert "qb3" not in protected  # a backup QB is fair game


def test_locked_players_are_never_dropped(client: TestClient) -> None:
    body = recommend(client, locked=["wr4"]).json()
    assert by_id(body["drops"]) == ["qb3"]
    reasons = {p["player"]["player_id"]: p["reason"] for p in body["protected"]}
    assert reasons["wr4"] == "locked"


def test_run_is_persisted_with_projection_version(client: TestClient, settings: Settings) -> None:
    body = recommend(client).json()
    assert settings.database_url is not None
    engine = create_engine(settings.database_url)
    try:
        with Session(engine) as session:
            (record,) = session.scalars(select(RecommendationRecord))
    finally:
        engine.dispose()
    assert str(record.id) == body["id"]
    assert record.projections_version == "abc123def456"
    assert record.request["roster"] == ROSTER
    assert record.response["projected_gain"] == body["projected_gain"]


def test_unknown_ids_are_422_and_listed(client: TestClient) -> None:
    r = recommend(client, free_agents=["nobody", "wr5"])
    assert r.status_code == 422
    assert "nobody" in r.json()["detail"]


def test_unprojected_slots_are_rejected_with_reason(client: TestClient) -> None:
    r = recommend(client, settings={"slots": {"QB": 1, "K": 1}})
    assert r.status_code == 422
    assert "not projected" in r.text


def test_three_moves_have_no_backtest_evidence(client: TestClient) -> None:
    body = recommend(client, max_moves=3).json()
    assert body["evidence"] is None
    assert len(body["adds"]) == len(body["drops"]) <= 3


def test_no_free_agents_means_no_moves(client: TestClient) -> None:
    body = recommend(client, free_agents=[]).json()
    assert body["adds"] == body["drops"] == []
    assert body["projected_gain"] == 0.0
    assert any("0 free agent" in c for c in body["caveats"])
