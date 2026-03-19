"""Model loading, scoring, and custom song prediction."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from src.config import MODEL_DIR, PRUNED_ROW_FEATURE_COLS, FILL_VALUES_FINAL, TOP_K
from src.data import make_feature_matrix


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pretrained_models() -> dict:
    """Load stage2 ranker and stage3 regressor from MODEL_DIR."""
    ranker = xgb.XGBRanker()
    ranker.load_model(str(MODEL_DIR / "stage2_country_ranker.json"))

    regressor = xgb.XGBRegressor()
    regressor.load_model(str(MODEL_DIR / "stage3_days_to_entry_regressor.json"))

    return {"ranker": ranker, "regressor": regressor}


# ---------------------------------------------------------------------------
# Scoring helpers (from notebook)
# ---------------------------------------------------------------------------

def normalize_scores(values: pd.Series) -> pd.Series:
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max > value_min:
        return (values - value_min) / (value_max - value_min)
    return pd.Series(np.full(len(values), 0.5), index=values.index)


def score_ranker(
    model, df: pd.DataFrame, feature_cols: list[str], fill_values: dict | pd.Series,
) -> pd.DataFrame:
    ordered = df.sort_values(["target_country"]).reset_index(drop=True)
    X = make_feature_matrix(ordered, feature_cols, fill_values)
    raw_scores = pd.Series(model.predict(X), index=ordered.index)
    scored = ordered[["target_country"]].copy()
    scored["score"] = normalize_scores(raw_scores)
    scored["raw_score"] = raw_scores.to_numpy()
    return scored


def transform_target(y, target_transform: str = "log1p") -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    if target_transform == "log1p":
        return np.log1p(y_arr)
    if target_transform == "sqrt":
        return np.sqrt(y_arr)
    return y_arr


def inverse_transform_target(y: np.ndarray, target_transform: str = "log1p") -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    if target_transform == "log1p":
        return np.expm1(y_arr)
    if target_transform == "sqrt":
        return np.square(y_arr)
    return y_arr


def score_regressor(
    model, df: pd.DataFrame, feature_cols: list[str], fill_values: dict | pd.Series,
    target_transform: str = "log1p",
) -> pd.DataFrame:
    X = make_feature_matrix(df, feature_cols, fill_values)
    preds = model.predict(X)
    preds = inverse_transform_target(preds, target_transform)
    scored = df[["target_country"]].copy()
    scored["predicted_days_to_entry"] = np.clip(preds, 1.0, 60.0)
    return scored


# ---------------------------------------------------------------------------
# Custom song prediction (NEW)
# ---------------------------------------------------------------------------

def predict_custom_song(
    models: dict,
    prediction_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    fill_values: dict | None = None,
    top_k: int = TOP_K,
    origin_countries: set[str] | None = None,
) -> dict:
    """Run ranker + regressor on a 62-row prediction DataFrame.
    origin_countries are excluded from the top-k (already charting there).
    Returns dict with 'top_k' results, 'all_scores', and 'timing_ms'.
    """
    if feature_cols is None:
        feature_cols = PRUNED_ROW_FEATURE_COLS
    if fill_values is None:
        fill_values = FILL_VALUES_FINAL

    t0 = time.time()

    # Stage 2: rank countries
    ranker = models["ranker"]
    X = make_feature_matrix(prediction_df, feature_cols, fill_values)
    raw_scores = pd.Series(ranker.predict(X), index=prediction_df.index)
    norm_scores = normalize_scores(raw_scores)

    # Stage 3: predict timing
    regressor = models["regressor"]
    timing_preds = regressor.predict(X)
    timing_preds = inverse_transform_target(timing_preds, "log1p")
    timing_preds = np.clip(timing_preds, 1.0, 60.0)

    elapsed_ms = (time.time() - t0) * 1000

    # Build results
    results = prediction_df[["target_country"]].copy()
    results["score"] = norm_scores.values
    results["raw_score"] = raw_scores.values
    results["predicted_days_to_entry"] = timing_preds
    results = results.sort_values("score", ascending=False).reset_index(drop=True)

    # Exclude origin countries (song already charts there)
    if origin_countries:
        candidates = results[~results["target_country"].isin(origin_countries)].copy()
    else:
        candidates = results.copy()
    candidates = candidates.reset_index(drop=True)
    candidates["predicted_rank"] = candidates.index + 1

    return {
        "top_k": candidates.head(top_k),
        "all_scores": candidates,
        "timing_ms": elapsed_ms,
    }
