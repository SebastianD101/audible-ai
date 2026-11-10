"""Application settings loaded from environment variables (and an optional ``.env``)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Unset means a local SQLite file (see backend/app/db/session.py).
    database_url: str | None = None
    projections_dir: Path = Path("ml/data/projections")
    rosters_dir: Path = Path("ml/data/raw/rosters_weekly")
    cors_origins: list[str] = ["http://localhost:3000"]
    sleeper_api_url: str = "https://api.sleeper.app/v1"


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
