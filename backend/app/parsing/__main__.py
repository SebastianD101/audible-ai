"""Parse one roster screenshot and print the result as JSON (manual smoke test).

Usage:
    uv run python -m backend.app.parsing path/to/screenshot.png [--season 2026]

This makes a real, billed API call using ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_MODEL`` from
the environment or ``.env``. The image is read into memory only and is not stored.
"""

import argparse
import logging
import sys
from pathlib import Path

from backend.app.config import get_settings
from backend.app.parsing.extract import ConfigurationError, ExtractionError
from backend.app.parsing.image import InvalidImageError
from backend.app.parsing.players import load_player_index
from backend.app.parsing.service import make_client, parse_roster_screenshot

DEFAULT_SEASON = 2026


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("image", type=Path, help="PNG, JPEG, WebP, or GIF screenshot")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON, help="roster season")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = get_settings()
    try:
        index = load_player_index(args.season)
        roster = parse_roster_screenshot(
            args.image.read_bytes(), make_client(settings), settings, index
        )
    except (FileNotFoundError, InvalidImageError, ConfigurationError, ExtractionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(roster.model_dump_json(indent=2))
    flagged = sum(p.needs_review for p in roster.players)
    print(f"{len(roster.players)} players, {flagged} flagged for review", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
