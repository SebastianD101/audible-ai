"""The real Sleeper API (run with ``-m network`` and ``SLEEPER_TEST_USERNAME`` set)."""

import os

import pytest

from backend.app.config import Settings
from backend.app.sleeper.client import SleeperClient

pytestmark = pytest.mark.network

USERNAME = os.environ.get("SLEEPER_TEST_USERNAME")


@pytest.mark.skipif(USERNAME is None, reason="set SLEEPER_TEST_USERNAME to a Sleeper username")
def test_real_user_leagues_and_rosters() -> None:
    assert USERNAME is not None
    client = SleeperClient(Settings(_env_file=None).sleeper_api_url)  # type: ignore[call-arg]
    user = client.user(USERNAME)
    leagues = client.leagues(user.user_id, 2026)
    assert leagues, "the test user should have a 2026 league"
    rosters = client.rosters(leagues[0].league_id)
    assert any(r.owner_id == user.user_id for r in rosters)
