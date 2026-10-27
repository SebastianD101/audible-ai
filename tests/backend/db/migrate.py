"""Run the real Alembic migrations against a given database URL."""

from pathlib import Path

from alembic import command
from alembic.config import Config

ALEMBIC_INI = Path(__file__).parents[3] / "backend" / "alembic.ini"


def alembic_config(url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.attributes["url"] = url
    config.attributes["skip_logging"] = True
    return config


def upgrade(url: str) -> None:
    command.upgrade(alembic_config(url), "head")


def downgrade(url: str) -> None:
    command.downgrade(alembic_config(url), "base")
