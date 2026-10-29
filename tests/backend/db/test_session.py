import pytest

from backend.app.config import Settings
from backend.app.db.session import DEFAULT_DATABASE_URL, database_url


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, DEFAULT_DATABASE_URL),
        ("sqlite:///x.db", "sqlite:///x.db"),
        (
            "postgres://u:p@host/db?sslmode=require",
            "postgresql+psycopg://u:p@host/db?sslmode=require",
        ),
        ("postgresql://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
    ],
)
def test_hosted_postgres_urls_use_psycopg_3(configured: str | None, expected: str) -> None:
    assert database_url(Settings(database_url=configured)) == expected
