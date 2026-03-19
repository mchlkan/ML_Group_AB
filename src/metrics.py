"""Evaluation functions extracted from the notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ranking_metrics(scored_df: pd.DataFrame, k: int = 5) -> tuple[dict, pd.DataFrame]:
    rows = []
    for track_id, group in scored_df.groupby("track_id", sort=False):
        group = group.sort_values(
            ["score", "tie_break"], ascending=[False, False],
        ).reset_index(drop=True)
        labels = group["did_enter_within_60d"].to_numpy(dtype=int)
        positives = int(labels.sum())
        top = group.head(k)
        top_labels = top["did_enter_within_60d"].to_numpy(dtype=int)
        hits = int(top_labels.sum())

        precision = hits / k
        recall = hits / positives if positives else np.nan
        hit_rate = float(hits > 0) if positives else np.nan

        discounts = np.log2(np.arange(2, len(top_labels) + 2))
        dcg = float(((2**top_labels - 1) / discounts).sum())
        ideal = np.sort(labels)[::-1][: len(top_labels)]
        idcg = float(((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))).sum())
        ndcg = dcg / idcg if idcg > 0 else np.nan

        ap_accum = 0.0
        running_hits = 0
        for rank, rel in enumerate(top_labels, start=1):
            if rel:
                running_hits += 1
                ap_accum += running_hits / rank
        map_k = ap_accum / min(positives, k) if positives else np.nan

        rows.append({
            "track_id": track_id,
            "positives": positives,
            "top_k_hits": hits,
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"hit_rate@{k}": hit_rate,
            f"ndcg@{k}": ndcg,
            f"map@{k}": map_k,
        })

    metric_df = pd.DataFrame(rows)
    positive_mask = metric_df["positives"] > 0
    summary = {
        "tracks": int(metric_df.shape[0]),
        "positive_tracks": int(positive_mask.sum()),
        f"precision@{k}": float(metric_df[f"precision@{k}"].mean()),
        f"recall@{k}": (
            float(metric_df.loc[positive_mask, f"recall@{k}"].mean())
            if positive_mask.any() else None
        ),
        f"hit_rate@{k}": (
            float(metric_df.loc[positive_mask, f"hit_rate@{k}"].mean())
            if positive_mask.any() else None
        ),
        f"ndcg@{k}": (
            float(metric_df.loc[positive_mask, f"ndcg@{k}"].mean())
            if positive_mask.any() else None
        ),
        f"map@{k}": (
            float(metric_df.loc[positive_mask, f"map@{k}"].mean())
            if positive_mask.any() else None
        ),
    }
    return summary, metric_df


def evaluate_ranked_candidates(scored_df: pd.DataFrame, k: int = 5) -> tuple[dict, pd.DataFrame]:
    ranking_all, track_metrics = ranking_metrics(scored_df, k=k)
    positive_track_metrics = track_metrics[track_metrics["positives"] > 0].copy()
    positive_summary = {
        "tracks": int(positive_track_metrics.shape[0]),
        f"recall@{k}": (
            float(positive_track_metrics[f"recall@{k}"].mean())
            if not positive_track_metrics.empty else None
        ),
        f"hit_rate@{k}": (
            float(positive_track_metrics[f"hit_rate@{k}"].mean())
            if not positive_track_metrics.empty else None
        ),
        f"ndcg@{k}": (
            float(positive_track_metrics[f"ndcg@{k}"].mean())
            if not positive_track_metrics.empty else None
        ),
        f"map@{k}": (
            float(positive_track_metrics[f"map@{k}"].mean())
            if not positive_track_metrics.empty else None
        ),
    }
    return {
        "ranking_all_tracks": ranking_all,
        "ranking_positive_tracks": positive_summary,
    }, track_metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

    abs_err = np.abs(y_true - y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "pct_within_3_days": float(np.mean(abs_err <= 3.0)),
        "pct_within_7_days": float(np.mean(abs_err <= 7.0)),
    }


def feature_category(name: str) -> str:
    if name.startswith("rank_") or name in {
        "track_in_viral50_at_obs", "candidate_count", "origin_country_count_at_obs",
    }:
        return "current_footprint"
    if name.startswith("artist_") or name == "multi_artist_flag":
        return "artist_history"
    if name.startswith("target_"):
        return "target_country_priors"
    if name in {
        "cultural_dist_min", "cultural_dist_missing", "same_language_flag",
        "song_lang_matches_target", "same_continent_flag", "neighbor_entered_count",
    }:
        return "origin_target_relationship"
    if name.endswith("_mean") or name.endswith("_max"):
        return "aggregates"
    if name.startswith("af_") or name in {
        "duration_ms", "explicit", "days_since_release", "is_friday_release",
    }:
        return "audio_track_metadata"
    if name.startswith("observation_"):
        return "temporal"
    return "other"
