"""End-to-end screenshot parsing: validate, extract, match, and flag rows for review.

Extraction is either the Claude vision model or local OCR (``Settings.parser``); everything
after it, matching and review flags, is shared, so both readers get the same safety net.
"""

from collections import Counter

import anthropic

from backend.app.config import Settings
from backend.app.parsing.extract import ConfigurationError, extract_roster
from backend.app.parsing.image import ValidatedImage, validate_image
from backend.app.parsing.match import match_player
from backend.app.parsing.ocr import OCR_MODEL_NAME, lines_to_roster, read_lines
from backend.app.parsing.players import PlayerIndex
from backend.app.parsing.schemas import ExtractedRoster, ParsedPlayer, ParsedRoster


def make_client(settings: Settings) -> anthropic.Anthropic | None:
    """Build an API client from settings, falling back to the SDK's own credential lookup.

    Returns None when screenshots are read with local OCR, which needs no client.
    """
    if settings.resolved_parser() == "ocr":
        return None
    if settings.anthropic_api_key is None:
        return anthropic.Anthropic()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())


def extract(
    image: ValidatedImage,
    client: anthropic.Anthropic | None,
    settings: Settings,
    index: PlayerIndex,
) -> tuple[ExtractedRoster, str]:
    """Read the roster with the configured reader; returns it and the reader's name.

    Raises:
        ConfigurationError: Claude is configured but no client was given.
        ConfigurationError, ExtractionError: see ``extract_roster`` and ``read_lines``.
    """
    if settings.resolved_parser() == "ocr":
        return lines_to_roster(read_lines(image), index), OCR_MODEL_NAME
    if client is None:
        raise ConfigurationError("PARSER=claude needs an Anthropic client.")
    roster = extract_roster(
        image, client, model=settings.anthropic_model, effort=settings.anthropic_effort
    )
    return roster, settings.anthropic_model or ""


def parse_roster_screenshot(
    data: bytes, client: anthropic.Anthropic | None, settings: Settings, index: PlayerIndex
) -> ParsedRoster:
    """Turn raw screenshot bytes into an ID-matched roster with review flags.

    A player needs review unless it was read clearly, matched uniquely, and no other row on
    the screenshot matched the same player.

    Raises:
        InvalidImageError: the upload is not an acceptable image.
        ConfigurationError, ExtractionError: see ``extract``.
    """
    image = validate_image(data, max_bytes=settings.max_upload_bytes)
    extracted, model = extract(image, client, settings, index)
    matches = [match_player(p, index) for p in extracted.players]
    counts = Counter(m.player_id for m in matches if m.player_id is not None)
    duplicates = {pid for pid, n in counts.items() if n > 1}

    players = [
        ParsedPlayer(
            extracted=p,
            match=m,
            needs_review=(
                p.legibility != "clear" or m.status != "matched" or m.player_id in duplicates
            ),
        )
        for p, m in zip(extracted.players, matches, strict=True)
    ]
    warnings = list(extracted.warnings)
    if duplicates:
        warnings.append("More than one row matched the same player; check the flagged rows.")
    return ParsedRoster(
        platform=extracted.platform,
        players=players,
        warnings=warnings,
        model=model,
    )
