"""The serve image hosts the static frontend on the same origin as the API."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.main import mount_frontend


def build_app(directory: Path | None) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/ping")
    def ping() -> dict[str, str]:
        return {"pong": "api"}

    mount_frontend(app, directory)
    return app


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    (tmp_path / "index.html").write_text("<html>audible ui</html>")
    (tmp_path / "_next").mkdir()
    (tmp_path / "_next" / "app.js").write_text("console.log(1)")
    return tmp_path


def test_serves_index_at_root(out_dir: Path) -> None:
    r = TestClient(build_app(out_dir)).get("/")
    assert r.status_code == 200
    assert "audible ui" in r.text


def test_serves_built_assets(out_dir: Path) -> None:
    r = TestClient(build_app(out_dir)).get("/_next/app.js")
    assert r.status_code == 200
    assert r.text == "console.log(1)"


def test_api_routes_take_priority(out_dir: Path) -> None:
    client = TestClient(build_app(out_dir))
    assert client.get("/api/ping").json() == {"pong": "api"}
    assert client.get("/health").json() == {"status": "ok"}


def test_unknown_paths_are_404_not_the_api(out_dir: Path) -> None:
    assert TestClient(build_app(out_dir)).get("/nope").status_code == 404


def test_nothing_is_mounted_when_unset() -> None:
    client = TestClient(build_app(None))
    assert client.get("/").status_code == 404
    assert client.get("/health").status_code == 200


def test_a_directory_without_a_build_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"index\.html"):
        build_app(tmp_path)
