"""Pipeline helpers extracted from the notebook."""

from __future__ import annotations

import pandas as pd


def add_predicted_rank(scored_df: pd.DataFrame) -> pd.DataFrame:
    ranked = scored_df.sort_values(
        ["track_id", "score", "target_new_entry_rate_30d"],
        ascending=[True, False, False],
    ).copy()
    ranked["predicted_rank"] = ranked.groupby("track_id").cumcount().add(1).astype(int)
    return ranked


def add_regression_predictions(
    row_scored_df: pd.DataFrame, reg_scored_df: pd.DataFrame,
) -> pd.DataFrame:
    return row_scored_df.merge(
        reg_scored_df[["track_id", "target_country", "predicted_days_to_entry"]],
        on=["track_id", "target_country"],
        how="left",
    )
