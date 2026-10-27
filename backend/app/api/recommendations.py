"""Add/drop recommendations for a reviewed roster."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.deps import PlayerIndexDep, ProjectionsDep
from backend.app.db.models import RecommendationRecord
from backend.app.db.session import get_session
from backend.app.recommend import (
    Recommendation,
    RecommendationRequest,
    UnknownPlayersError,
    build_recommendation,
)

router = APIRouter(prefix="/api", tags=["recommendations"])


class RecommendationResponse(Recommendation):
    """A stored recommendation."""

    id: uuid.UUID


@router.post("/recommendations")
def create_recommendation(
    request: RecommendationRequest,
    table: ProjectionsDep,
    index: PlayerIndexDep,
    session: Annotated[Session, Depends(get_session)],
) -> RecommendationResponse:
    """Recommend up to ``max_moves`` add/drop swaps for next week, and store the run."""
    try:
        rec = build_recommendation(request, table, index)
    except UnknownPlayersError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    record = RecommendationRecord(
        season=rec.season,
        week=rec.week,
        projections_version=rec.projections_version,
        request=request.model_dump(mode="json"),
        response=rec.model_dump(mode="json"),
    )
    session.add(record)
    session.flush()
    return RecommendationResponse(id=record.id, **rec.model_dump())
