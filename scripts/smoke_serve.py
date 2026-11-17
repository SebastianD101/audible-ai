"""Smoke-test a running serve image: the UI, the baked-in data, and a full recommendation.

The deploy workflow runs the image with Render's free-tier memory (512 MB) and then this
script, so an out-of-memory optimizer fails the job before anything ships. The recommendation
request is built from the committed projections, the same file the image serves, so it's
realistic in size: a full roster and a league's worth of free agents.

Usage: ``uv run python scripts/smoke_serve.py --base-url http://localhost:8000``
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl

PROJECTIONS_DIR = Path("ml/data/projections")
ROSTER_SIZE = 16
FREE_AGENTS = 200


def latest_projections(root: Path = PROJECTIONS_DIR) -> pl.DataFrame:
    """The newest committed week's projections."""
    files = sorted(root.glob("season=*/week=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no projections under {root}")
    return pl.read_parquet(files[-1])


def recommendation_request(projections: pl.DataFrame) -> dict[str, Any]:
    """A roster of the top projected players and the next tier as free agents."""
    ids = projections.sort("projection", descending=True)["player_id"].to_list()
    return {
        "roster": ids[:ROSTER_SIZE],
        "free_agents": ids[ROSTER_SIZE : ROSTER_SIZE + FREE_AGENTS],
        "max_moves": 2,
    }


def call(base_url: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, str]:
    """GET (or POST JSON) a path and return the status and body text."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base_url + path, data=data, method="POST" if data else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def wait_until_up(base_url: str, seconds: int = 60) -> None:
    """Poll ``/health`` until the service answers or time runs out."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if call(base_url, "/health")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError(f"{base_url}/health did not answer within {seconds}s")


def main() -> int:
    """Run every check; print each result and return non-zero if any fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    base = parser.parse_args().base_url.rstrip("/")

    wait_until_up(base)
    checks: list[tuple[str, str, dict[str, Any] | None, str]] = [
        ("ui", "/", None, "<html"),
        ("meta", "/api/projections/meta", None, '"season"'),
        ("search", "/api/players/search?q=allen", None, "["),
        ("recommend", "/api/recommendations", recommendation_request(latest_projections()), "adds"),
    ]
    failed = False
    for name, path, body, expect in checks:
        started = time.monotonic()
        status, text = call(base, path, body)
        ok = status == 200 and expect in text
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {status} in {time.monotonic() - started:.1f}s")
        if not ok:
            print(text[:500])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
