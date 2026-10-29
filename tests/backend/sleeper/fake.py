"""A fake Sleeper API on ``httpx.MockTransport``, with synthetic users and leagues."""

import json
from collections.abc import Callable
from typing import Any

import httpx

from backend.app.sleeper.client import SleeperClient
from tests.backend.api.world import SEASON, sleeper_id

BASE = "https://sleeper.test/v1"
USER_ID = "111"
OTHER_ID = "222"
LEAGUE_ID = "900"

POSITIONS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "WRRB_FLEX", "K", "DEF", "BN", "BN"]


def s(*player_ids: str) -> list[str]:
    """Sleeper IDs for world player IDs; team codes and "0" pass through."""
    return [p if p.isupper() or p == "0" else sleeper_id(p) for p in player_ids]


def league(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "league_id": LEAGUE_ID,
        "name": "Test League",
        "season": str(SEASON),  # Sleeper sends the season as a string
        "status": "in_season",
        "total_rosters": 2,
        "roster_positions": POSITIONS,
        "scoring_settings": {"rec": 1.0, "pass_td": 4.0},
        "sport": "nfl",
    }
    return data | overrides


def rosters() -> list[dict[str, Any]]:
    mine = {
        "roster_id": 1,
        "owner_id": USER_ID,
        "co_owners": None,
        # QB, RB, RB, WR, WR, TE, FLEX, W/R, K (empty), DEF
        "starters": s("qb1", "rb1", "rb2", "wr1", "wr3", "te1", "rb3", "wr2", "0", "BUF"),
        "players": [
            *s("qb1", "rb1", "rb2", "wr1", "wr3", "te1", "rb3", "wr2", "BUF", "rb4", "wr4"),
            "99999",  # not in the crosswalk
        ],
        "reserve": s("wr4"),
        "taxi": None,
    }
    theirs = {
        "roster_id": 2,
        "owner_id": OTHER_ID,
        "co_owners": [USER_ID + "0"],
        "starters": s("qb2"),
        "players": s("qb2", "te2"),
        "reserve": None,
        "taxi": None,
    }
    return [mine, theirs]


def routes() -> dict[str, Any]:
    return {
        "/user/someone": {"user_id": USER_ID, "username": "someone", "display_name": "Someone"},
        "/user/nobody": None,
        f"/user/{USER_ID}/leagues/nfl/{SEASON}": [league()],
        f"/league/{LEAGUE_ID}": league(),
        f"/league/{LEAGUE_ID}/rosters": rosters(),
    }


def transport(
    table: dict[str, Any] | None = None, calls: list[str] | None = None
) -> httpx.MockTransport:
    """Serve ``table`` (path -> JSON); exceptions in it are raised; unknown paths are 404."""
    table = routes() if table is None else table

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        if calls is not None:
            calls.append(request.url.raw_path.decode().removeprefix("/v1"))  # as sent
        if path not in table:
            return httpx.Response(404)
        body = table[path]
        if isinstance(body, Exception):
            raise body
        if isinstance(body, httpx.Response):
            return body
        # Encoded by hand: json=None would send an empty body, and Sleeper sends "null".
        return httpx.Response(200, content=json.dumps(body).encode())

    return httpx.MockTransport(handle)


def client(
    table: dict[str, Any] | None = None,
    calls: list[str] | None = None,
    clock: Callable[[], float] | None = None,
) -> SleeperClient:
    http = httpx.Client(transport=transport(table, calls))
    if clock is None:
        return SleeperClient(BASE, http=http)
    return SleeperClient(BASE, http=http, clock=clock)
