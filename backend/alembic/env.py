"""Alembic environment: migrate the database named by ``DATABASE_URL``.

A caller (tests) may pass a URL with ``config.attributes["url"]`` instead.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from backend.app.db.models import Base
from backend.app.db.session import database_url

config = context.config
if config.config_file_name is not None and not config.attributes.get("skip_logging"):
    fileConfig(config.config_file_name)

url = config.attributes.get("url") or database_url()


def run_offline() -> None:
    """Emit SQL to stdout instead of connecting."""
    context.configure(url=url, target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    """Connect and apply migrations."""
    engine = create_engine(url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
