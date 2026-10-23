import polars as pl
import pytest

from ml.backtest import predictors as pr


def features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["established", "early_season", "rookie"],
            "ppr_last3": [14.0, None, None],
            "ppr_season": [12.0, None, None],
            "prev_ppr": [9.0, 11.0, None],
        },
        schema_overrides={c: pl.Float64 for c in ["ppr_last3", "ppr_season", "prev_ppr"]},
    )


@pytest.mark.parametrize(
    ("predictor", "expected"),
    [(pr.Last3Baseline(), [14.0, 11.0, 0.0]), (pr.SeasonAvgBaseline(), [12.0, 11.0, 0.0])],
)
def test_baselines_fall_back_to_prior_season_then_zero(
    predictor: pr.Predictor, expected: list[float]
) -> None:
    predictor.fit(features())
    assert predictor.predict(features()).to_list() == expected


def test_baselines_satisfy_the_protocol_and_are_registered() -> None:
    for name, cls in pr.BASELINES.items():
        model: pr.Predictor = cls()
        assert model.name == name
