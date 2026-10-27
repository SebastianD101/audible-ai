"""The API contract is snapshotted so a change shows up in review next to the frontend types.

Regenerate after an intended change: ``UPDATE_OPENAPI=1 uv run pytest tests/backend/api``.
"""

import json
import os
from pathlib import Path

from backend.app.main import app

SNAPSHOT = Path(__file__).parent / "openapi.json"


def test_openapi_matches_snapshot() -> None:
    current = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_OPENAPI"):
        SNAPSHOT.write_text(current, encoding="utf-8")
    assert SNAPSHOT.read_text(encoding="utf-8") == current, (
        "API schema changed: update frontend/lib/api.ts, then rerun with UPDATE_OPENAPI=1"
    )
