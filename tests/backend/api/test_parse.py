from collections.abc import Callable

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_client
from backend.app.config import Settings
from backend.app.db.models import ParseRecord
from backend.app.main import app
from backend.app.parsing.schemas import ExtractedRoster
from tests.backend.parsing.frames import extracted
from tests.backend.parsing.test_extract import api_error, fake_client, response

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
ROSTER = ExtractedRoster(
    platform="sleeper",
    players=[
        extracted("Josh Allen", "QB", "BUF"),
        extracted("Puka Nacua", "WR", "LA"),
        extracted("B. Aubrey", "K", legibility="partial"),
    ],
    warnings=[],
)


Install = Callable[[object], None]


@pytest.fixture
def use_client(client: TestClient) -> Install:
    """Install a fake Anthropic client returning (or raising) ``result``."""

    def install(result: object) -> None:
        fake, _ = fake_client(result)
        app.dependency_overrides[get_client] = lambda: fake

    return install


def post(client: TestClient, data: bytes = PNG, kind: str = "roster") -> httpx2.Response:
    return client.post(
        "/api/parse", files={"file": ("shot.png", data, "image/png")}, data={"kind": kind}
    )


def records(settings: Settings) -> list[ParseRecord]:
    assert settings.database_url is not None
    engine = create_engine(settings.database_url)
    try:
        with Session(engine) as session:
            return list(session.scalars(select(ParseRecord)))
    finally:
        engine.dispose()


def test_parse_matches_players_attaches_projections_and_persists(
    client: TestClient, settings: Settings, use_client: Install
) -> None:
    use_client(response(ROSTER))
    r = post(client, kind="free_agents")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "free_agents"
    ids = [p["match"]["player_id"] for p in body["roster"]["players"]]
    assert ids[:2] == ["qb1", "wr1"]
    assert body["players"]["qb1"]["projection"] == 24.0
    assert body["roster"]["players"][2]["needs_review"] is True  # partial legibility

    (record,) = records(settings)
    assert str(record.id) == body["id"]
    assert record.kind == "free_agents"
    assert len(record.players) == 3


def test_record_never_contains_image_bytes(
    client: TestClient, settings: Settings, use_client: Install
) -> None:
    use_client(response(ROSTER))
    post(client)
    (record,) = records(settings)
    stored = repr((record.players, record.warnings, record.platform, record.model))
    assert "PNG" not in stored


def test_oversized_upload_is_413(
    client: TestClient, settings: Settings, use_client: Install
) -> None:
    use_client(response(ROSTER))
    settings.max_upload_bytes = 16
    r = post(client)
    assert r.status_code == 413
    assert records(settings) == []


def test_non_image_is_422(client: TestClient, use_client: Install) -> None:
    use_client(response(ROSTER))
    r = post(client, data=b"not an image at all")
    assert r.status_code == 422
    assert "PNG" in r.json()["detail"]


def test_missing_model_is_503(client: TestClient, settings: Settings, use_client: Install) -> None:
    use_client(response(ROSTER))
    settings.anthropic_model = None
    r = post(client)
    assert r.status_code == 503
    assert "ANTHROPIC_MODEL" in r.json()["detail"]


def test_rate_limit_is_503_with_retry_after(client: TestClient, use_client: Install) -> None:
    use_client(api_error(anthropic.RateLimitError, 429))
    r = post(client)
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "30"


def test_non_retryable_failure_is_502(client: TestClient, use_client: Install) -> None:
    use_client(response(None, stop_reason="refusal"))
    r = post(client)
    assert r.status_code == 502
