import os
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from backend.app.db.models import Base, ParseRecord, RecommendationRecord
from tests.backend.db.migrate import downgrade, upgrade

TABLES = {"parse_records", "recommendation_records"}


def check_migrations_match_models_and_round_trip(url: str) -> None:
    upgrade(url)
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        assert diff == [], f"models and migrations disagree: {diff}"

        with Session(engine) as session, session.begin():
            session.add(
                ParseRecord(
                    kind="roster",
                    platform="sleeper",
                    model="m",
                    players=[{"name": "A", "match": {"status": "matched"}}],
                    warnings=["w"],
                )
            )
            session.add(
                RecommendationRecord(
                    season=2026, week=8, projections_version="v", request={"a": 1}, response={}
                )
            )
        with Session(engine) as session:
            record = session.scalars(select(ParseRecord)).one()
            assert record.players[0]["match"]["status"] == "matched"
            assert record.created_at is not None
    finally:
        engine.dispose()

    downgrade(url)
    engine = create_engine(url)
    try:
        assert not TABLES & set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_sqlite_migrations_match_models_and_round_trip(tmp_path: Path) -> None:
    check_migrations_match_models_and_round_trip(f"sqlite:///{(tmp_path / 't.db').as_posix()}")


@pytest.mark.postgres
def test_postgres_migrations_match_models_and_round_trip() -> None:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    check_migrations_match_models_and_round_trip(url)


def test_no_table_stores_image_bytes() -> None:
    for table in Base.metadata.tables.values():
        for column in table.columns:
            assert "image" not in column.name
            assert column.type.python_type is not bytes
