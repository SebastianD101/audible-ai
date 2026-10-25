from typing import Any

import polars as pl

from ml.features import kickoff as ko
from ml.features import load
from ml.features import spine as sp
from ml.features.asof import Week
from tests.ml.features import synthetic as syn
from tests.ml.features.lookahead import assert_no_lookahead


def spine(as_of: Week) -> pl.DataFrame:
    return sp.spine(syn.stats(), syn.rosters(), sp.fixtures(syn.games()), as_of)


def features(as_of: Week) -> pl.DataFrame:
    return ko.kickoff_features(spine(as_of), load.game_lines(syn.schedule()), syn.injuries(), as_of)


def row(df: pl.DataFrame, player: str) -> dict[str, Any]:
    return df.filter(pl.col("player_id") == player).row(0, named=True)


def test_game_lines_sign_the_spread_for_each_team() -> None:
    lines = load.game_lines(syn.schedule()).filter(pl.col("season") == 2023, pl.col("week") == 1)
    by_team = dict(zip(lines["team"], lines["team_spread"], strict=True))
    assert by_team == {"AAA": 3.0, "BBB": -3.0, "CCC": 3.0, "DDD": -3.0}
    assert set(lines["total_line"]) == {44.0}


def test_implied_team_total_favors_the_favorite() -> None:
    df = features(Week(2023, 3))  # AAA hosts BBB, home favored by 3, total 44.
    assert row(df, "wr1")["ko_implied_team_total"] == 23.5
    assert row(df, "wr2")["ko_implied_team_total"] == 20.5
    assert row(df, "wr1")["ko_team_spread"] == 3.0


def test_uses_the_target_weeks_own_injury_report() -> None:
    # Waiver-time features see week 2's Questionable; kickoff features see week 3's Out.
    wr1 = row(features(Week(2023, 3)), "wr1")
    assert wr1["ko_report_severity"] == 3
    assert wr1["ko_practice_level"] is None
    assert row(features(Week(2023, 2)), "wr1")["ko_practice_level"] == 1
    assert row(features(Week(2023, 3)), "rb1")["ko_report_severity"] is None


def test_ignores_everything_after_the_target_week() -> None:
    target = Week(2023, 3)
    spine_rows = spine(target)

    def at_target(lines: pl.DataFrame, injuries: pl.DataFrame, as_of: Week) -> pl.DataFrame:
        assert as_of == Week(target.season, target.week + 1)
        return ko.kickoff_features(spine_rows, lines, injuries, target)

    assert_no_lookahead(
        at_target,
        {"lines": load.game_lines(syn.schedule()), "injuries": syn.injuries()},
        Week(target.season, target.week + 1),
    )


def test_available_through_includes_the_week_itself() -> None:
    df = pl.DataFrame({"season": [2023, 2023, 2023], "week": [2, 3, 4]})
    assert ko.available_through(df, Week(2023, 3))["week"].to_list() == [2, 3]


def test_repeated_keys_do_not_duplicate_spine_rows() -> None:
    as_of = Week(2023, 3)
    lines = load.game_lines(syn.schedule())
    doubled = ko.kickoff_features(
        spine(as_of), pl.concat([lines, lines]), pl.concat([syn.injuries(), syn.injuries()]), as_of
    )
    assert doubled.height == spine(as_of).height
    assert row(doubled, "wr1")["ko_report_severity"] == 3
