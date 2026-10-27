"""ORM tables. JSON columns keep the schema portable (Postgres in deployment, SQLite in tests).

Screenshots are never stored: a parse record keeps only the extracted text and matches.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Current time, timezone-aware."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every table."""


class ParseRecord(Base):
    """One parsed screenshot: what was read and how each row matched."""

    __tablename__ = "parse_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    kind: Mapped[str] = mapped_column(String(16))
    platform: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    players: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    warnings: Mapped[list[str]] = mapped_column(JSON)


class RecommendationRecord(Base):
    """One recommendation request and its response, tied to a projections version."""

    __tablename__ = "recommendation_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    season: Mapped[int] = mapped_column(Integer)
    week: Mapped[int] = mapped_column(Integer)
    projections_version: Mapped[str] = mapped_column(String(64))
    request: Mapped[dict[str, Any]] = mapped_column(JSON)
    response: Mapped[dict[str, Any]] = mapped_column(JSON)
