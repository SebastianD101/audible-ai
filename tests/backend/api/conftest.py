from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, get_settings
from backend.app.main import app
from tests.backend.api.world import write_world
from tests.backend.db.migrate import upgrade


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    rosters_dir, projections_dir = write_world(tmp_path)
    url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
    upgrade(url)
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        parser="claude",
        anthropic_api_key=None,
        anthropic_model="test-model",
        database_url=url,
        projections_dir=projections_dir,
        rosters_dir=rosters_dir,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
