from typing import Any

import polars as pl
import pytest

from ml.features import context
from ml.features.asof import Week
from ml.features.spine import fixtures
from tests.ml.features import synthetic as syn
from tests.ml.features.lookahead import assert_no_lookahead


def features(as_of: Week) -> pl.DataFrame:
    return context.context_features(fixtures(syn.games()), syn.games(), as_of)


def team(df: pl.DataFrame, name: str) -> dict[str, Any]:
    return df.filter(pl.col("team") == name).row(0, named=True)


def test_venue_and_rest_come_from_the_week_w_fixture() -> None:
    out = features(Week(2023, 3))
    assert set(out["team"]) == {"AAA", "BBB", "CCC", "DDD"}
    aaa, bbb = team(out, "AAA"), team(out, "BBB")
    assert (aaa["is_home"], bbb["is_home"]) == (True, False)
    assert (aaa["rest_diff"], aaa["div_game"], aaa["indoor"]) == (0, False, False)


def test_bye_teams_have_no_context_row() -> None:
    assert set(features(Week(2023, 4))["team"]) == {"CCC", "DDD"}


def test_scoring_rates_shrink_toward_prior_season() -> None:
    # Home teams score 24 and allow 17. In 2022 AAA was home in weeks 1, 2, 3, 5 -> 24/17.
    # In 2023 before week 3, AAA played two home games (24, 17).
    aaa = team(features(Week(2023, 3)), "AAA")
    assert aaa["points_for_pg"] == pytest.approx(24.0)
    assert aaa["points_against_pg"] == pytest.approx(17.0)
    # BBB: 2022 away in weeks 1, 3, 5 (17/24), home in week 2 (24/17); 2023 so far: away wk 1
    # (17/24), home wk 2 (24/17).
    bbb = team(features(Week(2023, 3)), "BBB")
    prev_pf = (17 * 3 + 24) / 4
    assert bbb["points_for_pg"] == pytest.approx((2 * 20.5 + 4 * prev_pf) / 6)
    assert aaa["implied_total_proxy"] == pytest.approx(
        (aaa["points_for_pg"] + bbb["points_against_pg"]) / 2
    )


def test_first_season_falls_back_to_league_average() -> None:
    out = features(Week(2022, 2))
    # One game played each; no 2021 data, so prior = league average of 2022 week 1 (20.5).
    aaa = team(out, "AAA")
    assert aaa["points_for_pg"] == pytest.approx((24 + 4 * 20.5) / 5)


def test_context_does_not_look_ahead() -> None:
    assert_no_lookahead(
        context.context_features,
        {"fixtures": fixtures(syn.games()), "games": syn.games()},
        Week(2023, 3),
        known_in_advance=frozenset({"fixtures"}),
    )
