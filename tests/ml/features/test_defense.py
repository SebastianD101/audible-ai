import polars as pl
import pytest

from ml.features import defense
from ml.features.asof import Week
from tests.ml.features import synthetic as syn
from tests.ml.features.lookahead import assert_no_lookahead


def wr_line(season: int, week: int, offense: str, opp: str, points: float) -> dict[str, object]:
    return {
        "player_id": f"{offense}-wr",
        "position": "WR",
        "season": season,
        "week": week,
        "game_id": f"{season}_{week}_{offense}_{opp}",
        "team": offense,
        "opponent": opp,
        "fantasy_points_ppr": points,
    }


def sos_league() -> pl.DataFrame:
    """Strong WR offense O1 (30/game) and weak O2 (10/game) facing defenses D1 and D2.

    D1 faces O1 twice and gives up a big week-3 game (36); D2 faces O2 twice.
    """
    rows = [
        wr_line(2023, 1, "O1", "D1", 30.0),
        wr_line(2023, 1, "O2", "D2", 10.0),
        wr_line(2023, 2, "O1", "D2", 30.0),
        wr_line(2023, 2, "O2", "D1", 10.0),
        wr_line(2023, 3, "O1", "D1", 36.0),
        wr_line(2023, 3, "O2", "D2", 10.0),
    ]
    return pl.DataFrame(rows).with_columns(pl.col("season", "week").cast(pl.Int32))


def rating(df: pl.DataFrame, defense_team: str, position: str = "WR") -> dict[str, object]:
    return df.filter(pl.col("opponent") == defense_team, pl.col("position") == position).row(
        0, named=True
    )


def test_sos_adjustment_separates_schedule_from_defense_quality() -> None:
    out = defense.defense_features(sos_league(), Week(2023, 4))
    d1, d2 = rating(out, "D1"), rating(out, "D2")
    # Raw allowed: D1 76/3 vs D2 50/3, mostly because D1 faced O1 twice.
    assert d1["dvp_raw_allowed"] == pytest.approx(76 / 3)
    assert d2["dvp_raw_allowed"] == pytest.approx(50 / 3)
    # Residuals: D1 [-3, 0, +6] -> +1; D2 [0, -3, 0] -> -1.
    assert d1["dvp_season"] == pytest.approx(1.0)
    assert d2["dvp_season"] == pytest.approx(-1.0)
    assert d1["dvp_games"] == 3
    # No prior season: shrink toward 0 with K=4.
    assert d1["dvp_adj"] == pytest.approx(3 * 1.0 / (3 + defense.SHRINK_GAMES))


def test_positions_without_stat_lines_count_as_zero_production() -> None:
    out = defense.defense_features(sos_league(), Week(2023, 4))
    assert rating(out, "D1", "TE")["dvp_raw_allowed"] == 0.0


def test_prior_season_rating_carries_into_new_season() -> None:
    last_year = sos_league().with_columns(pl.lit(2022, dtype=pl.Int32).alias("season"))
    out = defense.defense_features(last_year, Week(2023, 2))
    d1 = rating(out, "D1")
    assert d1["dvp_games"] == 0
    assert d1["dvp_prev"] == pytest.approx(1.0)
    assert d1["dvp_adj"] == pytest.approx(defense.PRIOR_SEASON_WEIGHT * 1.0)


def test_consistent_scorer_yields_zero_residual() -> None:
    # qb1 scores 20 every game, so every defense's QB rating is exactly 0.
    out = defense.defense_features(syn.stats(), Week(2023, 5))
    assert out.filter(pl.col("position") == "QB")["dvp_season"].abs().max() == pytest.approx(0.0)


def test_defense_does_not_look_ahead() -> None:
    assert_no_lookahead(defense.defense_features, {"stats": syn.stats()}, Week(2023, 3))
