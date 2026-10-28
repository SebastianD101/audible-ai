import httpx
import pytest

from backend.app.sleeper.client import SleeperNotFoundError, SleeperUnavailableError
from tests.backend.api.world import SEASON
from tests.backend.sleeper.fake import LEAGUE_ID, USER_ID, client


def test_user_leagues_and_rosters_parse() -> None:
    c = client()
    user = c.user("someone")
    assert user.user_id == USER_ID
    (league,) = c.leagues(user.user_id, SEASON)
    assert league.season == SEASON  # sent as a string
    assert [r.roster_id for r in c.rosters(LEAGUE_ID)] == [1, 2]


@pytest.mark.parametrize("username", ["nobody", "missing"])
def test_null_or_404_is_not_found(username: str) -> None:
    with pytest.raises(SleeperNotFoundError):
        client().user(username)


@pytest.mark.parametrize(
    "body",
    [httpx.Response(500), httpx.Response(200, text="<html>"), httpx.ConnectTimeout("slow")],
)
def test_errors_and_timeouts_are_unavailable(body: object) -> None:
    with pytest.raises(SleeperUnavailableError):
        client({"/user/someone": body}).user("someone")


def test_responses_are_cached_briefly() -> None:
    calls: list[str] = []
    now = [0.0]
    c = client(calls=calls, clock=lambda: now[0])
    c.user("someone")
    c.user("someone")
    assert len(calls) == 1
    now[0] = 61.0
    c.user("someone")
    assert len(calls) == 2


def test_not_found_is_not_cached() -> None:
    calls: list[str] = []
    c = client(calls=calls)
    for _ in range(2):
        with pytest.raises(SleeperNotFoundError):
            c.user("nobody")
    assert len(calls) == 2


def test_username_is_escaped_into_one_path_segment() -> None:
    calls: list[str] = []
    with pytest.raises(SleeperNotFoundError):
        client(calls=calls).user("a/../b")
    assert calls == ["/user/a%2F..%2Fb"]
