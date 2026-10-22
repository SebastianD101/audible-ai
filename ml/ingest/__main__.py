"""Run every nflverse raw-layer loader in order.

Usage:
    uv run python -m ml.ingest --start-season 2018 --end-season 2025 [--force]

Each dataset fails independently: a failure is logged and the remaining datasets still run.
The process exits non-zero if any dataset failed. The player ID crosswalk runs last so it can
report how well it maps the stored snap counts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from ml.ingest import (
    common,
    injuries,
    players,
    rosters_weekly,
    schedules,
    snap_counts,
    weekly_stats,
)

logger = logging.getLogger(__name__)

SPECS: tuple[common.DatasetSpec, ...] = (
    weekly_stats.SPEC,
    schedules.SPEC,
    rosters_weekly.SPEC,
    injuries.SPEC,
    snap_counts.SPEC,
)


def run_all(
    seasons: list[int],
    *,
    force: bool = False,
    specs: Sequence[common.DatasetSpec] = SPECS,
    ingest_crosswalk: Callable[..., object] = players.ingest_players,
) -> list[str]:
    """Ingest every dataset for ``seasons`` and return the names of those that failed."""
    failed: list[str] = []
    for spec in specs:
        try:
            common.ingest_seasons(spec, seasons, force=force)
        except Exception:
            logger.exception("%s: ingest failed", spec.name)
            failed.append(spec.name)
    try:
        ingest_crosswalk(force=force)
    except Exception:
        logger.exception("players: ingest failed")
        failed.append("players")
    return failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=common.first_line(__doc__))
    parser.add_argument("--start-season", type=int, default=common.DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=common.DEFAULT_END_SEASON)
    parser.add_argument("--force", action="store_true", help="re-download existing data")
    return common.parse_season_args(parser, argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    common.configure_logging()
    args = parse_args(argv)
    seasons = list(range(args.start_season, args.end_season + 1))
    failed = run_all(seasons, force=args.force)
    if failed:
        logger.error("failed datasets: %s", ", ".join(failed))
        sys.exit(1)
    logger.info("all datasets ingested for %d-%d", args.start_season, args.end_season)


if __name__ == "__main__":
    main()
