"""Application settings loaded from environment variables (and an optional ``.env``)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Secrets are ``SecretStr`` so they never appear in logs or reprs."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str | None = None
    # Effort for screenshot extraction; set explicitly because model defaults differ.
    anthropic_effort: Literal["low", "medium", "high"] = "medium"
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, gt=0, le=5 * 1024 * 1024)
    # Unset means a local SQLite file (see backend/app/db/session.py).
    database_url: str | None = None
    projections_dir: Path = Path("ml/data/projections")
    rosters_dir: Path = Path("ml/data/raw/rosters_weekly")
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
