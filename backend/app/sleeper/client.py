"""A small, cached client for Sleeper's read-only public API.

The API needs no key and is free for non-commercial use, with a guideline of under 1000 calls a
minute. Responses are cached for a minute so repeat clicks don't hit it again. Sleeper answers
``200 null`` for an unknown user or league, which is reported as ``SleeperNotFoundError``.
"""

import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

from backend.app.sleeper.schemas import SleeperLeague, SleeperRoster, SleeperUser

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10.0
CACHE_SECONDS = 60.0


class SleeperError(RuntimeError):
    """Base class for Sleeper API problems."""


class SleeperNotFoundError(SleeperError):
    """The user or league doesn't exist."""


class SleeperUnavailableError(SleeperError):
    """Sleeper couldn't be reached or returned an error; a retry may work."""


class SleeperClient:
    """Typed GETs against the Sleeper API with a short in-memory cache."""

    def __init__(
        self,
        base_url: str,
        http: httpx.Client | None = None,
        cache_seconds: float = CACHE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Use ``http`` if given (tests pass a mock transport), else a fresh client."""
        self._http = http or httpx.Client(timeout=TIMEOUT_SECONDS)
        self._base = base_url.rstrip("/")
        self._ttl = cache_seconds
        self._clock = clock
        self._cache: dict[str, tuple[float, Any]] = {}

    def _get(self, path: str, what: str) -> Any:
        now = self._clock()
        hit = self._cache.get(path)
        if hit is not None and now - hit[0] < self._ttl:
            return hit[1]
        try:
            response = self._http.get(f"{self._base}{path}")
        except httpx.HTTPError as e:
            logger.warning("Sleeper request failed: %s", type(e).__name__)
            raise SleeperUnavailableError("Couldn't reach Sleeper; try again shortly.") from e
        if response.status_code == 404:
            raise SleeperNotFoundError(f"Sleeper has no {what}.")
        if response.status_code >= 400:
            logger.warning("Sleeper returned HTTP %d for %s", response.status_code, what)
            raise SleeperUnavailableError(f"Sleeper returned an error ({response.status_code}).")
        try:
            body = response.json()
        except ValueError as e:
            raise SleeperUnavailableError("Sleeper returned an unreadable response.") from e
        if body is None:
            raise SleeperNotFoundError(f"Sleeper has no {what}.")
        self._cache[path] = (now, body)
        return body

    def user(self, username: str) -> SleeperUser:
        """Look up an account by username (or user ID)."""
        data = self._get(f"/user/{quote(username, safe='')}", f"user named {username!r}")
        return SleeperUser.model_validate(data)

    def leagues(self, user_id: str, season: int) -> list[SleeperLeague]:
        """The user's NFL leagues for one season."""
        data = self._get(f"/user/{quote(user_id, safe='')}/leagues/nfl/{season}", "leagues")
        return [SleeperLeague.model_validate(item) for item in data]

    def league(self, league_id: str) -> SleeperLeague:
        """One league's settings."""
        data = self._get(f"/league/{quote(league_id, safe='')}", f"league {league_id}")
        return SleeperLeague.model_validate(data)

    def rosters(self, league_id: str) -> list[SleeperRoster]:
        """Every team's roster in a league."""
        path = f"/league/{quote(league_id, safe='')}/rosters"
        return [SleeperRoster.model_validate(item) for item in self._get(path, "rosters")]
