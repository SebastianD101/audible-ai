"""Read the raw Parquet layer into clean, consistently keyed frames for feature building.

Everything here is a deterministic cleanup with no time logic: the as-of cutoff is applied
later, inside each feature function. Cleanups (each documented in ``docs/decisions.md``):

- Regular season only; fantasy features and targets are regular-season.
- Team codes normalized (``OAK`` -> ``LV`` for 2018-2019, where sources disagree).
- Player IDs are gsis IDs everywhere, under the column name ``player_id``. Snap counts are
  mapped from PFR IDs through the crosswalk; unmapped rows (<0.4% of offensive rows) dropped.
- Duplicate keys left in the raw layer are resolved here: injuries keep the most severe
  designation; rosters keep the row whose name matches the crosswalk.
- Whitespace-only strings (seen in ``practice_status``) become null.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

DEFAULT_RAW_DIR = Path("ml/data/raw")
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")
TEAM_ALIASES = {"OAK": "LV"}
# Ordinal severity of the game designation; higher is less likely to play.
REPORT_SEVERITY = {"Note": 0, "Questionable": 1, "Doubtful": 2, "Out": 3}

STAT_COLUMNS = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "target_share",
    "air_yards_share",
    "wopr",
    "fantasy_points",
    "fantasy_points_ppr",
]

# Raw columns each cleaner reads, per dataset.
RAW_COLUMNS = {
    "player_stats": [
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
        "season_type",
        "game_id",
        "team",
        "opponent_team",
        *STAT_COLUMNS,
    ],
    "schedules": [
        "season",
        "week",
        "game_type",
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_rest",
        "away_rest",
        "div_game",
        "roof",
    ],
    "rosters_weekly": [
        "gsis_id",
        "full_name",
        "season",
        "week",
        "game_type",
        "team",
        "position",
        "status",
    ],
    "injuries": [
        "gsis_id",
        "season",
        "week",
        "game_type",
        "team",
        "report_status",
        "practice_status",
    ],
    "snap_counts": [
        "pfr_player_id",
        "season",
        "week",
        "game_type",
        "game_id",
        "team",
        "offense_snaps",
        "offense_pct",
    ],
}


@dataclass(frozen=True)
class RawData:
    """Clean frames consumed by the feature modules (all regular season)."""

    stats: pl.DataFrame
    games: pl.DataFrame
    rosters: pl.DataFrame
    injuries: pl.DataFrame
    snaps: pl.DataFrame


def normalize_teams(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Map historical team codes to current ones in ``columns``."""
    return df.with_columns(pl.col(columns).replace(TEAM_ALIASES))


def blank_to_null(column: str) -> pl.Expr:
    """Return ``column`` with whitespace-only strings replaced by null."""
    stripped = pl.col(column).str.strip_chars()
    return pl.when(stripped == "").then(None).otherwise(stripped).alias(column)


def read_seasons(raw_dir: Path, dataset: str, columns: list[str]) -> pl.DataFrame:
    """Read ``columns`` from every season file of a dataset.

    Projecting up front matters: unused columns drift in dtype between seasons (e.g. roster
    ``height`` is Float64 in some years and Int32 in others) and would fail a full scan.
    """
    return (
        pl.scan_parquet(
            raw_dir / dataset / "season=*.parquet",
            missing_columns="insert",
            extra_columns="ignore",
        )
        .select(columns)
        .collect()
    )


def clean_stats(raw: pl.DataFrame) -> pl.DataFrame:
    """Regular-season player-game stat lines with a known player, all positions.

    All positions are kept so team totals (e.g. team carries) are complete; the spine restricts
    to fantasy positions.
    """
    df = raw.filter(pl.col("season_type") == "REG", pl.col("player_id").is_not_null()).select(
        "player_id",
        pl.col("player_display_name").alias("name"),
        "position",
        "season",
        "week",
        "game_id",
        "team",
        pl.col("opponent_team").alias("opponent"),
        *STAT_COLUMNS,
    )
    return normalize_teams(df, ["team", "opponent"])


def team_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Reshape regular-season games to one row per team per game.

    Columns: season, week, game_id, team, opponent, is_home, rest, div_game, roof,
    points_for, points_against. Scores are outcomes: only feature functions that apply the
    as-of cutoff may read them.
    """
    games = normalize_teams(
        schedules.filter(pl.col("game_type") == "REG"), ["home_team", "away_team"]
    )
    common = ["season", "week", "game_id", "div_game", "roof"]
    home = games.select(
        *common,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        pl.col("home_rest").alias("rest"),
        pl.col("home_score").alias("points_for"),
        pl.col("away_score").alias("points_against"),
    )
    away = games.select(
        *common,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        pl.col("away_rest").alias("rest"),
        pl.col("away_score").alias("points_for"),
        pl.col("home_score").alias("points_against"),
    )
    return pl.concat([home, away]).sort("season", "week", "team")


def clean_rosters(raw: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Regular-season weekly rosters at fantasy positions, one row per player-week.

    Where upstream lists two people under one gsis ID in a week, the row whose name matches
    the crosswalk's ``display_name`` is kept.
    """
    df = raw.filter(
        pl.col("game_type") == "REG",
        pl.col("gsis_id").is_not_null(),
        pl.col("position").is_in(FANTASY_POSITIONS),
    ).select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("full_name").alias("name"),
        "season",
        "week",
        "team",
        "position",
        "status",
    )
    names = players.select(pl.col("gsis_id").alias("player_id"), "display_name")
    return (
        normalize_teams(df, ["team"])
        .join(names, on="player_id", how="left")
        .sort(pl.col("name") != pl.col("display_name"), nulls_last=True)
        .unique(["player_id", "season", "week"], keep="first", maintain_order=True)
        .drop("display_name")
        .sort("season", "week", "player_id")
    )


def clean_injuries(raw: pl.DataFrame) -> pl.DataFrame:
    """Regular-season injury reports, one row per player-week (most severe designation)."""
    df = raw.filter(pl.col("game_type") == "REG", pl.col("gsis_id").is_not_null()).select(
        pl.col("gsis_id").alias("player_id"),
        "season",
        "week",
        "team",
        "report_status",
        blank_to_null("practice_status"),
        pl.col("report_status").replace_strict(REPORT_SEVERITY, default=None).alias("severity"),
    )
    return (
        normalize_teams(df, ["team"])
        .sort("severity", descending=True, nulls_last=True)
        .unique(["player_id", "season", "week"], keep="first", maintain_order=True)
        .sort("season", "week", "player_id")
    )


def clean_snaps(raw: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Regular-season offensive snap counts keyed by gsis ID via the PFR crosswalk."""
    crosswalk = players.filter(pl.col("pfr_id").is_not_null()).select(
        pl.col("pfr_id").alias("pfr_player_id"), pl.col("gsis_id").alias("player_id")
    )
    df = (
        raw.filter(pl.col("game_type") == "REG")
        .join(crosswalk, on="pfr_player_id", how="inner")
        .select("player_id", "season", "week", "game_id", "team", "offense_snaps", "offense_pct")
    )
    return normalize_teams(df, ["team"])


def load_raw(raw_dir: Path = DEFAULT_RAW_DIR) -> RawData:
    """Read and clean every raw dataset."""
    players = pl.read_parquet(raw_dir / "players" / "players.parquet")
    return RawData(
        stats=clean_stats(read_seasons(raw_dir, "player_stats", RAW_COLUMNS["player_stats"])),
        games=team_games(read_seasons(raw_dir, "schedules", RAW_COLUMNS["schedules"])),
        rosters=clean_rosters(
            read_seasons(raw_dir, "rosters_weekly", RAW_COLUMNS["rosters_weekly"]), players
        ),
        injuries=clean_injuries(read_seasons(raw_dir, "injuries", RAW_COLUMNS["injuries"])),
        snaps=clean_snaps(
            read_seasons(raw_dir, "snap_counts", RAW_COLUMNS["snap_counts"]), players
        ),
    )
