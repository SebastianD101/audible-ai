from typing import Any

import polars as pl
import pytest

from ml.features import availability as av
from ml.features import spine as sp
from ml.features.asof import Week
from tests.ml.features import synthetic as syn
from tests.ml.features.lookahead import assert_no_lookahead


def inputs(as_of: Week, snaps: pl.DataFrame | None = None) -> dict[str, pl.DataFrame]:
    return {
        "spine": sp.spine(syn.stats(), syn.rosters(), sp.fixtures(syn.games()), as_of),
        "stats": syn.stats(),
        "snaps": syn.snaps() if snaps is None else snaps,
        "rosters": syn.rosters(),
        "injuries": syn.injuries(),
        "games": syn.games(),
    }


def features(as_of: Week, snaps: pl.DataFrame | None = None) -> pl.DataFrame:
    return av.availability_features(**inputs(as_of, snaps), as_of=as_of)


def row(df: pl.DataFrame, player: str) -> dict[str, Any]:
    return df.filter(pl.col("player_id") == player).row(0, named=True)


def test_missed_last_team_game_after_a_bye_uses_the_last_game_played() -> None:
    # AAA's last game before week 5 is week 3 (week 4 bye); wr1 missed it, listed Out.
    wr1 = row(features(Week(2023, 5)), "wr1")
    assert wr1["missed_last_game"] is True
    assert wr1["games_missed_last3"] == 1
    assert (wr1["last_report_status"], wr1["last_report_severity"]) == ("Out", 3)
    assert wr1["roster_status_last_week"] == "ACT"


def test_designation_from_last_game_week_includes_practice_level() -> None:
    wr1 = row(features(Week(2023, 3)), "wr1")
    assert wr1["missed_last_game"] is False
    assert (wr1["last_report_status"], wr1["last_practice_level"]) == ("Questionable", 1)


def test_healthy_player_has_no_flags() -> None:
    rb1 = row(features(Week(2023, 5)), "rb1")
    assert (rb1["missed_last_game"], rb1["left_last_game_early"], rb1["games_missed_last3"]) == (
        False,
        False,
        0,
    )
    assert rb1["last_report_status"] is None


def test_teammate_out_vacates_target_share() -> None:
    out = features(Week(2023, 5))
    # wr1 (AAA WR, 0.25 target share) missed AAA's last game.
    rb1, wr1 = row(out, "rb1"), row(out, "wr1")
    assert rb1["vacated_target_share_team"] == pytest.approx(0.25)
    assert rb1["vacated_target_share_pos"] == pytest.approx(0.0)
    assert wr1["vacated_target_share_team"] == pytest.approx(0.0)


def test_leaving_last_game_early_counts_as_out() -> None:
    hurt = syn.snaps().with_columns(
        pl.when((pl.col("player_id") == "rb1") & (pl.col("season") == 2023) & (pl.col("week") == 2))
        .then(0.1)
        .otherwise(pl.col("offense_pct"))
        .alias("offense_pct")
    )
    out = features(Week(2023, 3), snaps=hurt)
    # Week-2 share 0.1 vs a norm of 0.5 from week 1 alone.
    assert row(out, "rb1")["left_last_game_early"] is True
    assert row(out, "wr1")["vacated_rush_share_team"] == pytest.approx(1.0)


def test_left_early_needs_an_earlier_game_for_a_norm() -> None:
    hurt = syn.snaps().with_columns(
        pl.when((pl.col("player_id") == "rb1") & (pl.col("season") == 2023) & (pl.col("week") == 1))
        .then(0.1)
        .otherwise(pl.col("offense_pct"))
        .alias("offense_pct")
    )
    assert row(features(Week(2023, 2), snaps=hurt), "rb1")["left_last_game_early"] is False


def test_availability_does_not_look_ahead() -> None:
    as_of = Week(2023, 5)
    assert_no_lookahead(
        av.availability_features, inputs(as_of), as_of, known_in_advance=frozenset({"spine"})
    )
