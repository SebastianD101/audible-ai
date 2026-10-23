import polars as pl

from ml.features import spine as sp
from ml.features.asof import Week
from tests.ml.features import synthetic as syn
from tests.ml.features.lookahead import assert_no_lookahead


def build(as_of: Week) -> pl.DataFrame:
    return sp.spine(syn.stats(), syn.rosters(), sp.fixtures(syn.games()), as_of)


def test_fixtures_drop_results() -> None:
    assert not set(sp.OUTCOME_COLUMNS) & set(sp.fixtures(syn.games()).columns)


def test_week_one_has_no_spine() -> None:
    assert build(Week(2023, 1)).is_empty()
    assert build(Week(2023, 1)).columns == sp.SPINE_COLUMNS


def test_spine_lists_current_players_with_their_week_w_opponent() -> None:
    out = build(Week(2023, 3)).sort("player_id")
    assert out.select("player_id", "team", "opponent").rows() == [
        ("qb1", "CCC", "DDD"),
        ("rb1", "AAA", "BBB"),
        ("te1", "DDD", "CCC"),
        ("wr1", "AAA", "BBB"),
        ("wr2", "BBB", "AAA"),
    ]


def test_player_from_last_season_only_is_excluded() -> None:
    assert "old" not in build(Week(2023, 2))["player_id"].to_list()


def test_bye_week_teams_are_dropped() -> None:
    assert set(build(Week(2023, 4))["team"]) == {"CCC", "DDD"}


def test_prior_week_roster_team_beats_last_stat_line_team() -> None:
    traded = syn.rosters().with_columns(
        pl.when((pl.col("player_id") == "wr2") & (pl.col("week") == 2))
        .then(pl.lit("CCC"))
        .otherwise(pl.col("team"))
        .alias("team")
    )
    out = sp.spine(syn.stats(), traded, sp.fixtures(syn.games()), Week(2023, 3))
    assert out.filter(pl.col("player_id") == "wr2")["team"].item() == "CCC"


def test_spine_falls_back_to_stat_lines_when_roster_is_missing() -> None:
    no_rosters = syn.rosters().clear()
    out = sp.spine(syn.stats(), no_rosters, sp.fixtures(syn.games()), Week(2023, 3))
    assert set(out["player_id"]) == {"wr1", "rb1", "wr2", "qb1", "te1"}


def test_practice_squad_players_need_a_stat_line() -> None:
    extra = pl.DataFrame(
        {
            "player_id": ["ps1", "cut1"],
            "name": ["PS1", "CUT1"],
            "season": [2023, 2023],
            "week": [2, 2],
            "team": ["AAA", "AAA"],
            "position": ["WR", "WR"],
            "status": ["DEV", "CUT"],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    rosters = pl.concat([syn.rosters(), extra])
    out = sp.spine(syn.stats(), rosters, sp.fixtures(syn.games()), Week(2023, 3))
    assert not {"ps1", "cut1"} & set(out["player_id"])


def test_spine_does_not_look_ahead() -> None:
    assert_no_lookahead(
        sp.spine,
        {"stats": syn.stats(), "rosters": syn.rosters(), "fixtures": sp.fixtures(syn.games())},
        Week(2023, 3),
        known_in_advance=frozenset({"fixtures"}),
    )


def test_labels_score_played_games_and_zero_for_missed_games() -> None:
    labelled = sp.add_labels(build(Week(2023, 3)), syn.stats())
    wr1 = labelled.filter(pl.col("player_id") == "wr1").row(0, named=True)
    qb1 = labelled.filter(pl.col("player_id") == "qb1").row(0, named=True)
    assert (wr1["label_ppr"], wr1["label_std"], wr1["label_played"]) == (0.0, 0.0, False)
    assert (qb1["label_ppr"], qb1["label_std"], qb1["label_played"]) == (20.0, 17.0, True)
