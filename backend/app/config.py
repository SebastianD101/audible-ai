"""Application settings loaded from environment variables (and an optional ``.env``)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Secrets are ``SecretStr`` so they never appear in logs or reprs."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Screenshot reader: "claude" (vision model, billed), "ocr" (local and free), or "auto"
    # (Claude when a key and model are configured, else OCR).
    parser: Literal["auto", "claude", "ocr"] = "auto"
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
    sleeper_api_url: str = "https://api.sleeper.app/v1"

    def resolved_parser(self) -> Literal["claude", "ocr"]:
        """The screenshot reader to use, with ``auto`` resolved."""
        if self.parser != "auto":
            return self.parser
        return "claude" if self.anthropic_api_key and self.anthropic_model else "ocr"


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
