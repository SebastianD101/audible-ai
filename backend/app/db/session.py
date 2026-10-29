"""Engine and per-request sessions.

``DATABASE_URL`` selects the database (Postgres via docker-compose). When it's unset, a local
SQLite file is used so the API runs without Docker. Either way the schema comes from Alembic
migrations, never ``create_all``, so dev and deployment can't drift.
"""

from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import Settings, get_settings

DEFAULT_DATABASE_URL = "sqlite:///audible.db"


def database_url(settings: Settings | None = None) -> str:
    """The configured database URL, or the local SQLite default.

    Hosted Postgres providers (Neon, Fly, Railway) hand out ``postgres://`` or
    ``postgresql://`` URLs, which SQLAlchemy maps to psycopg2; this app installs psycopg 3.
    """
    url = (settings or get_settings()).database_url or DEFAULT_DATABASE_URL
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


@lru_cache
def engine_for(url: str) -> Engine:
    """One pooled engine per URL."""
    return create_engine(url, pool_pre_ping=True)


def get_session(settings: Annotated[Settings, Depends(get_settings)]) -> Iterator[Session]:
    """FastAPI dependency: a session that commits on success and rolls back on error."""
    factory = sessionmaker(engine_for(database_url(settings)), expire_on_commit=False)
    with factory.begin() as session:
        yield session
