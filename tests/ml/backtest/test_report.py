import polars as pl

from ml.backtest import report
from ml.backtest.harness import evaluation_weeks, walk_forward
from ml.backtest.predictors import Last3Baseline, SeasonAvgBaseline
from ml.backtest.scopes import attach_scopes
from tests.ml.backtest.frames import feature_table


def predictions() -> pl.DataFrame:
    df = feature_table()
    preds = walk_forward(
        df, [Last3Baseline(), SeasonAvgBaseline()], evaluation_weeks(df, [2021, 2022])
    )
    return attach_scopes(preds, df)


def test_table_has_values_for_every_slice_and_diffs_only_vs_reference() -> None:
    table = report.metrics_table(predictions(), n_boot=20)
    values = table.filter(pl.col("kind") == "value")
    assert values.height == 2 * 2 * 5 * 3  # models x scopes x positions x metrics
    diffs = table.filter(pl.col("kind") == "diff")
    assert set(diffs["model"]) == {"season_avg"}
    assert diffs.height == 2 * 5 * 3


def test_values_report_sample_sizes() -> None:
    table = report.metrics_table(predictions(), n_boot=20)
    overall = table.filter(
        pl.col("kind") == "value", pl.col("scope") == "all", pl.col("position") == "ALL"
    )
    assert set(overall["n_weeks"]) == {6}
    assert set(overall["n_rows"]) == {6 * 12}


def test_markdown_has_a_table_per_scope_and_kind() -> None:
    md = report.to_markdown(report.metrics_table(predictions(), n_boot=20))
    assert md.count("| Model | Position |") == 4
    assert "### Difference vs `last3`: `relevant` rows" in md
    assert "week-block bootstrap" in md


def test_diff_pairs_compare_every_model_to_each_baseline_once() -> None:
    assert report.diff_pairs(["last3", "season_avg", "lgbm"], report.REFERENCES) == [
        ("season_avg", "last3"),
        ("lgbm", "last3"),
        ("lgbm", "season_avg"),
    ]
    assert report.diff_pairs(["lgbm"], report.REFERENCES) == []


def test_markdown_has_a_diff_block_per_reference() -> None:
    preds = predictions()
    third = preds.filter(pl.col("model") == "season_avg").with_columns(
        pl.lit("other").alias("model"), pl.col("prediction") + 1
    )
    md = report.to_markdown(report.metrics_table(pl.concat([preds, third]), n_boot=20))
    assert "### Difference vs `last3`: `all` rows" in md
    assert "### Difference vs `season_avg`: `all` rows" in md
