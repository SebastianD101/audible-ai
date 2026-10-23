"""A tiny, fully known league for feature tests (shapes match ``ml.features.load``).

Teams AAA, BBB, CCC, DDD; seasons 2022-2023; weeks 1-5.
- Odd weeks: AAA hosts BBB, CCC hosts DDD. Even weeks 2: AAA hosts CCC, BBB hosts DDD.
- Week 4: only CCC hosts DDD, so AAA and BBB are on bye.
Players (gsis-style IDs):
- ``wr1`` AAA WR. Scores ``10 + week`` PPR. Misses 2023 week 3: no stat line, injury report
  ``Out`` that week and ``Questionable`` in week 2.
- ``rb1`` AAA RB. Scores 8 every game.
- ``wr2`` BBB WR. Scores 5 every game.
- ``qb1`` CCC QB. Scores 20 every game.
- ``te1`` DDD TE. Scores 6 every game.
- ``old`` AAA WR. Plays 2022 only (gone in 2023).
"""

import polars as pl

from ml.features import load

SEASONS = (2022, 2023)
WEEKS = range(1, 6)
TEAM_OF = {"wr1": "AAA", "rb1": "AAA", "wr2": "BBB", "qb1": "CCC", "te1": "DDD", "old": "AAA"}
POSITION_OF = {"wr1": "WR", "rb1": "RB", "wr2": "WR", "qb1": "QB", "te1": "TE", "old": "WR"}
FLAT_PPR = {"rb1": 8.0, "wr2": 5.0, "qb1": 20.0, "te1": 6.0, "old": 4.0}


def matchups(week: int) -> list[tuple[str, str]]:
    """Return (home, away) pairs for a week."""
    if week == 4:
        return [("CCC", "DDD")]
    if week % 2:
        return [("AAA", "BBB"), ("CCC", "DDD")]
    return [("AAA", "CCC"), ("BBB", "DDD")]


def schedule() -> pl.DataFrame:
    """Raw-schedule-shaped frame; home teams win by 7."""
    rows = [
        (s, w, f"{s}_{w:02d}_{away}_{home}", home, away)
        for s in SEASONS
        for w in WEEKS
        for home, away in matchups(w)
    ]
    n = len(rows)
    return pl.DataFrame(
        {
            "season": [r[0] for r in rows],
            "week": [r[1] for r in rows],
            "game_type": ["REG"] * n,
            "game_id": [r[2] for r in rows],
            "home_team": [r[3] for r in rows],
            "away_team": [r[4] for r in rows],
            "home_score": [24] * n,
            "away_score": [17] * n,
            "home_rest": [7] * n,
            "away_rest": [7] * n,
            "div_game": [0] * n,
            "roof": ["outdoors"] * n,
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )


def games() -> pl.DataFrame:
    """Team-game rows via the production reshaping."""
    return load.team_games(schedule())


def plays(player: str, season: int, week: int) -> bool:
    """Whether a player has a stat line that week."""
    team_games = games().filter(pl.col("season") == season, pl.col("week") == week)
    if TEAM_OF[player] not in team_games["team"].to_list():
        return False
    if player == "old":
        return season == 2022
    return not (player == "wr1" and season == 2023 and week == 3)


def ppr(player: str, week: int) -> float:
    """PPR points for a game the player plays."""
    return 10.0 + week if player == "wr1" else FLAT_PPR[player]


def stats() -> pl.DataFrame:
    """Player-game stat lines shaped like ``load.clean_stats``."""
    g = games()
    rows = []
    for s in SEASONS:
        for w in WEEKS:
            for p, team in TEAM_OF.items():
                if not plays(p, s, w):
                    continue
                game = g.filter(pl.col("season") == s, pl.col("week") == w, pl.col("team") == team)
                rows.append(
                    {
                        "player_id": p,
                        "name": p.upper(),
                        "position": POSITION_OF[p],
                        "season": s,
                        "week": w,
                        "game_id": game["game_id"].item(),
                        "team": team,
                        "opponent": game["opponent"].item(),
                        **dict.fromkeys(load.STAT_COLUMNS, 0.0),
                        "targets": 6.0 if POSITION_OF[p] in ("WR", "TE") else 2.0,
                        "carries": 12.0 if POSITION_OF[p] == "RB" else 0.0,
                        "target_share": 0.25 if POSITION_OF[p] in ("WR", "TE") else 0.05,
                        "fantasy_points_ppr": ppr(p, w),
                        "fantasy_points": ppr(p, w) - 3.0,
                    }
                )
    return pl.DataFrame(rows).with_columns(pl.col("season", "week").cast(pl.Int32))


def rosters() -> pl.DataFrame:
    """Weekly rosters shaped like ``load.clean_rosters`` (everyone ACT on their team)."""
    rows = [
        {
            "player_id": p,
            "name": p.upper(),
            "season": s,
            "week": w,
            "team": team,
            "position": POSITION_OF[p],
            "status": "ACT",
        }
        for s in SEASONS
        for w in WEEKS
        for p, team in TEAM_OF.items()
        if not (p == "old" and s == 2023)
    ]
    return pl.DataFrame(rows).with_columns(pl.col("season", "week").cast(pl.Int32))


def injuries() -> pl.DataFrame:
    """Injury reports shaped like ``load.clean_injuries``."""
    return pl.DataFrame(
        {
            "player_id": ["wr1", "wr1"],
            "season": [2023, 2023],
            "week": [2, 3],
            "team": ["AAA", "AAA"],
            "report_status": ["Questionable", "Out"],
            "practice_status": ["Limited Participation in Practice", None],
            "severity": [1, 3],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32, "severity": pl.Int64},
    )


def snaps() -> pl.DataFrame:
    """Offensive snap shares for every stat line (WRs 0.8, others 0.5)."""
    return stats().select(
        "player_id",
        "season",
        "week",
        "game_id",
        "team",
        pl.lit(60.0).alias("offense_snaps"),
        pl.when(pl.col("position") == "WR").then(0.8).otherwise(0.5).alias("offense_pct"),
    )


def world() -> load.RawData:
    """Every synthetic frame bundled like ``load.load_raw``."""
    return load.RawData(
        stats=stats(), games=games(), rosters=rosters(), injuries=injuries(), snaps=snaps()
    )
