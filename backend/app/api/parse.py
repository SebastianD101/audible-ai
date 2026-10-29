"""Screenshot upload: parse a roster or a waiver list into ID-matched players.

The image is read into memory with a hard size cap, validated by its bytes, sent to the vision
model, and discarded. Only the extracted text and matches are stored.
"""

import logging
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.api.deps import ClientDep, PlayerIndexDep, ProjectionsDep, SettingsDep
from backend.app.db.models import ParseRecord
from backend.app.db.session import get_session
from backend.app.parsing.extract import ConfigurationError, ExtractionError
from backend.app.parsing.image import InvalidImageError
from backend.app.parsing.players import PlayerIndex
from backend.app.parsing.schemas import ParsedRoster
from backend.app.parsing.service import parse_roster_screenshot
from backend.app.projections import ProjectionTable
from backend.app.search import PlayerHit, hits_for

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["parse"])

ParseKind = Literal["roster", "free_agents"]
RETRY_AFTER_SECONDS = 30


class ParseResponse(BaseModel):
    """A parsed screenshot plus display info for every matched or candidate player."""

    id: uuid.UUID
    kind: ParseKind
    roster: ParsedRoster
    players: dict[str, PlayerHit]


def store_parse(
    session: Session,
    kind: ParseKind,
    roster: ParsedRoster,
    index: PlayerIndex,
    table: ProjectionTable,
) -> ParseResponse:
    """Persist a parsed roster and return it with display info for every player it mentions."""
    record = ParseRecord(
        kind=kind,
        platform=roster.platform,
        model=roster.model,
        players=[p.model_dump(mode="json") for p in roster.players],
        warnings=roster.warnings,
    )
    session.add(record)
    session.flush()
    ids = {c.player_id for p in roster.players for c in p.match.candidates}
    ids |= {p.match.player_id for p in roster.players if p.match.player_id}
    logger.info("parsed %s: %d players (record %s)", kind, len(roster.players), record.id)
    return ParseResponse(
        id=record.id, kind=kind, roster=roster, players=hits_for(index, table, ids)
    )


def read_capped(upload: UploadFile, max_bytes: int) -> bytes:
    """Read at most ``max_bytes``; reject anything larger without reading it all.

    Raises:
        HTTPException: 413 if the upload is over the cap.
    """
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"Image is over the {max_bytes // (1024 * 1024)} MB limit.",
        )
    return data


@router.post("/parse")
def parse_screenshot(
    settings: SettingsDep,
    table: ProjectionsDep,
    index: PlayerIndexDep,
    client: ClientDep,
    session: Annotated[Session, Depends(get_session)],
    file: Annotated[UploadFile, File(description="PNG, JPEG, WebP, or GIF screenshot")],
    kind: Annotated[ParseKind, Form()] = "roster",
) -> ParseResponse:
    """Parse one screenshot. Rows flagged ``needs_review`` must be confirmed by the user."""
    data = read_capped(file, settings.max_upload_bytes)
    try:
        roster = parse_roster_screenshot(data, client, settings, index)
    except InvalidImageError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    except ConfigurationError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    except ExtractionError as e:
        if e.retryable:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                str(e),
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            ) from e
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e
    return store_parse(session, kind, roster, index, table)
