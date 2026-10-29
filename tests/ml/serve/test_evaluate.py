from datetime import UTC, datetime

import polars as pl
import pytest

from ml.serve import evaluate
from ml.serve.evaluate import Excluded, scorable_weeks

KICKOFF = datetime(2026, 10, 30, 0, 15, tzinfo=UTC)


def test_first_kickoffs_convert_eastern_time_to_utc() -> None:
    schedules = pl.DataFrame(
        {
            "week": [8, 8, 8],
            "game_type": ["REG", "REG", "REG"],
            "gameday": ["2026-11-01", "2026-10-29", "2026-11-02"],
            "gametime": ["13:00", "20:15", "20:15"],
        }
    )
    assert evaluate.first_kickoffs(schedules) == {8: KICKOFF}


def test_final_weeks_need_every_game_scored() -> None:
    games = pl.DataFrame(
        {"season": [2026] * 4, "week": [7, 7, 8, 8], "points_for": [20, 17, 24, None]}
    )
    assert evaluate.final_weeks(games, 2026) == {7}


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("2026-10-27T23:13:47+00:00", ([8], [])),
        (
            "2026-10-30T00:15:00+00:00",
            ([], [Excluded(8, "generated after the week's first kickoff")]),
        ),
    ],
)
def test_projections_made_after_kickoff_are_excluded(
    generated: str, expected: tuple[list[int], list[Excluded]]
) -> None:
    manifests: dict[int, dict[str, object]] = {8: {"generated_at_utc": generated}}
    assert scorable_weeks(manifests, {8}, {8: KICKOFF}) == expected


def test_unfinished_weeks_wait_silently() -> None:
    manifests: dict[int, dict[str, object]] = {8: {"generated_at_utc": "2026-10-27T00:00:00+00:00"}}
    assert scorable_weeks(manifests, set(), {8: KICKOFF}) == ([], [])


def projections(n_qb: int = 30) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [f"qb{i:02d}" for i in range(n_qb)],
            "season": [2026] * n_qb,
            "week": [8] * n_qb,
            "position": ["QB"] * n_qb,
            "hold_value": [float(n_qb - i) for i in range(n_qb)],
            "projection": [float(n_qb - i) + 0.5 for i in range(n_qb)],
        }
    )


def test_relevant_keeps_the_top_n_by_pre_week_score() -> None:
    flags = evaluate.relevant(projections().sample(fraction=1.0, shuffle=True, seed=1))
    assert flags.sum() == 24  # QB limit


def test_evaluate_scores_both_models_and_their_difference() -> None:
    p = projections()
    labeled = p.with_columns(
        pl.col("hold_value").alias("label_ppr"), evaluate.relevant(p).alias("relevant")
    )
    results = evaluate.evaluate(labeled)
    assert results["relevant.QB.season_avg"]["rmse"].point == 0.0  # it was exactly right
    assert results["relevant.QB.lgbm"]["rmse"].point == pytest.approx(0.5)
    assert results["relevant.QB.lgbm-season_avg"]["rmse"].point == pytest.approx(0.5)
    assert results["relevant.ALL.lgbm"]["rmse"].n_rows == 24
    assert "relevant.RB.lgbm" not in results


def test_report_says_when_nothing_is_scorable_and_when_the_sample_is_tiny() -> None:
    assert "nothing to score" in evaluate.report(2026, [], [], {})
    p = projections()
    labeled = p.with_columns(
        pl.col("hold_value").alias("label_ppr"), evaluate.relevant(p).alias("relevant")
    )
    text = evaluate.report(2026, [8], [], evaluate.evaluate(labeled))
    assert "far too few weeks" in text
    assert "| QB | LightGBM minus season avg | 0.50" in text
