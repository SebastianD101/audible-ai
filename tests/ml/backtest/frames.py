"""Small feature tables shaped like ``ml.features.build`` output, for backtest tests."""

import polars as pl

from ml.features.spine import LABEL_COLUMNS, SPINE_COLUMNS

POSITIONS = {"q": "QB", "r": "RB", "w": "WR", "t": "TE"}


def feature_table(
    seasons: tuple[int, ...] = (2021, 2022), weeks: range = range(2, 5), per_position: int = 3
) -> pl.DataFrame:
    """Players ``{pos}{i}`` score ``10 * i + week`` PPR; ``ppr_season`` is ``10 * i``.

    Higher-numbered players are better, so pre-week rankings are known in advance.
    """
    rows = []
    for season in seasons:
        for week in weeks:
            for code, position in POSITIONS.items():
                for i in range(1, per_position + 1):
                    ppr = float(10 * i + week)
                    rows.append(
                        {
                            "player_id": f"{code}{i}",
                            "name": f"{code}{i}",
                            "position": position,
                            "season": season,
                            "week": week,
                            "team": "AAA",
                            "opponent": "BBB",
                            "game_id": f"{season}_{week:02d}_BBB_AAA",
                            "ppr_last3": float(10 * i + 1),
                            "ppr_season": float(10 * i),
                            "prev_ppr": float(10 * i - 1),
                            "label_ppr": ppr,
                            "label_std": ppr - 2,
                            "label_played": True,
                        }
                    )
    df = pl.DataFrame(rows).with_columns(pl.col("season", "week").cast(pl.Int32))
    assert set(SPINE_COLUMNS + LABEL_COLUMNS) <= set(df.columns)
    return df
