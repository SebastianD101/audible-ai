"""Turn walk-forward predictions into a tidy metrics table and a Markdown report."""

from __future__ import annotations

import polars as pl

from ml.backtest.metrics import (
    DEFAULT_BOOTSTRAP,
    DEFAULT_SEED,
    Estimate,
    bootstrap,
    paired_bootstrap,
)
from ml.backtest.scopes import SCOPES, in_scope

POSITIONS = ("ALL", "QB", "RB", "WR", "TE")
REFERENCE_MODEL = "last3"


def slice_rows(preds: pl.DataFrame, model: str, scope: str, position: str) -> pl.DataFrame:
    """One model's predictions restricted to a scope and position ("ALL" keeps every position)."""
    rows = in_scope(preds.filter(pl.col("model") == model), scope)
    return rows if position == "ALL" else rows.filter(pl.col("position") == position)


def estimate_rows(
    estimates: dict[str, Estimate], model: str, scope: str, position: str, kind: str
) -> list[dict[str, object]]:
    """Flatten estimates into table rows."""
    return [
        {
            "model": model,
            "scope": scope,
            "position": position,
            "kind": kind,
            "metric": e.metric,
            "point": e.point,
            "lo": e.lo,
            "hi": e.hi,
            "n_weeks": e.n_weeks,
            "n_rows": e.n_rows,
        }
        for e in estimates.values()
    ]


def metrics_table(
    preds: pl.DataFrame,
    *,
    reference: str = REFERENCE_MODEL,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> pl.DataFrame:
    """Metrics with intervals per (model, scope, position), plus paired diffs vs ``reference``.

    ``kind`` is ``"value"`` for a model's own metric and ``"diff"`` for model minus reference
    (negative RMSE/MAE diffs and positive Spearman diffs favour the model).
    """
    models = preds["model"].unique(maintain_order=True).to_list()
    rows: list[dict[str, object]] = []
    for model in models:
        for scope in SCOPES:
            for position in POSITIONS:
                sub = slice_rows(preds, model, scope, position)
                if sub.is_empty():
                    continue
                rows += estimate_rows(bootstrap(sub, n_boot, seed), model, scope, position, "value")
                if model != reference and reference in models:
                    ref = slice_rows(preds, reference, scope, position)
                    diff = paired_bootstrap(sub, ref, n_boot, seed)
                    rows += estimate_rows(diff, model, scope, position, "diff")
    return pl.DataFrame(rows)


def fmt(point: float, lo: float, hi: float) -> str:
    """Format a point estimate with its interval."""
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def to_markdown(table: pl.DataFrame, reference: str = REFERENCE_MODEL) -> str:
    """Render one table per scope (values) and one per scope (diffs vs ``reference``)."""
    lines: list[str] = []
    for kind, title in (("value", "Metrics"), ("diff", f"Difference vs `{reference}`")):
        part = table.filter(pl.col("kind") == kind)
        if part.is_empty():
            continue
        for scope in SCOPES:
            scoped = part.filter(pl.col("scope") == scope)
            if scoped.is_empty():
                continue
            lines += [
                f"### {title}: `{scope}` rows",
                "",
                "| Model | Position | Weeks | Rows | RMSE | MAE | Spearman (weekly) |",
                "| --- | --- | ---: | ---: | --- | --- | --- |",
            ]
            for (model, position), group in scoped.group_by(
                "model", "position", maintain_order=True
            ):
                cells = {
                    r["metric"]: fmt(r["point"], r["lo"], r["hi"])
                    for r in group.iter_rows(named=True)
                }
                first = group.row(0, named=True)
                lines.append(
                    f"| {model} | {position} | {first['n_weeks']} | {first['n_rows']:,} | "
                    f"{cells['rmse']} | {cells['mae']} | {cells['spearman']} |"
                )
            lines.append("")
    lines.append(
        "Intervals: 95% week-block bootstrap. "
        "Diffs are model minus reference on the same resampled weeks."
    )
    return "\n".join(lines) + "\n"
