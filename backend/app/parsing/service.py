"""End-to-end screenshot parsing: validate, extract, match, and flag rows for review."""

from collections import Counter

import anthropic

from backend.app.config import Settings
from backend.app.parsing.extract import extract_roster
from backend.app.parsing.image import validate_image
from backend.app.parsing.match import match_player
from backend.app.parsing.players import PlayerIndex
from backend.app.parsing.schemas import ParsedPlayer, ParsedRoster


def make_client(settings: Settings) -> anthropic.Anthropic:
    """Build an API client from settings, falling back to the SDK's own credential lookup."""
    if settings.anthropic_api_key is None:
        return anthropic.Anthropic()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())


def parse_roster_screenshot(
    data: bytes, client: anthropic.Anthropic, settings: Settings, index: PlayerIndex
) -> ParsedRoster:
    """Turn raw screenshot bytes into an ID-matched roster with review flags.

    A player needs review unless it was read clearly, matched uniquely, and no other row on
    the screenshot matched the same player.

    Raises:
        InvalidImageError: the upload is not an acceptable image.
        ConfigurationError, ExtractionError: see ``extract_roster``.
    """
    image = validate_image(data, max_bytes=settings.max_upload_bytes)
    extracted = extract_roster(
        image, client, model=settings.anthropic_model, effort=settings.anthropic_effort
    )
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
        model=settings.anthropic_model or "",
    )
