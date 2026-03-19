"""Demo Mode: browse pre-computed test set predictions."""

from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st

from src.config import COUNTRY_LIST, EVAL_DIR, TEST_PATH, TOP_K, V2_DATA_DIR, country_to_rank_col


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_parquet(EVAL_DIR / "test_predictions.parquet")


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_parquet(EVAL_DIR / "feature_importance.parquet")


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    return pd.read_parquet(EVAL_DIR / "model_comparison.parquet")


@st.cache_data
def load_pipeline_evaluation() -> pd.DataFrame:
    return pd.read_parquet(EVAL_DIR / "pipeline_evaluation.parquet")


@st.cache_data
def load_origin_footprints() -> pd.DataFrame:
    """Load rank columns from v3_features test split (one row per track)."""
    con = duckdb.connect()
    rank_cols = [country_to_rank_col(c) for c in COUNTRY_LIST]
    cols = ", ".join(["track_id"] + rank_cols)
    df = con.execute(
        f"SELECT {cols} FROM read_parquet('{TEST_PATH.as_posix()}') "
        f"GROUP BY track_id, {', '.join(rank_cols)}"
    ).fetchdf()
    con.close()
    return df


def _get_origin_footprint(track_id: str, track_df: pd.DataFrame) -> pd.DataFrame:
    """Get the countries where a track was charting at observation time."""
    footprints = load_origin_footprints()
    row = footprints[footprints["track_id"] == track_id]
    if row.empty:
        return pd.DataFrame(columns=["Country", "Rank"])
    row = row.iloc[0]
    entries = []
    for country in COUNTRY_LIST:
        rank_col = country_to_rank_col(country)
        rank = row.get(rank_col, 0)
        if rank and rank > 0:
            entries.append({"Country": country, "Rank": int(rank)})
    return pd.DataFrame(entries).sort_values("Rank")


@st.cache_data
def load_track_names() -> pd.DataFrame:
    """Load track_id → title + artist mapping from the v2 dataset."""
    con = duckdb.connect()
    v2 = f"read_parquet('{V2_DATA_DIR.as_posix()}/*/*.parquet')"
    names = con.execute(
        f"SELECT DISTINCT track_id, FIRST(title) AS title, FIRST(artist) AS artist "
        f"FROM {v2} GROUP BY track_id"
    ).fetchdf()
    con.close()
    return names


@st.cache_data
def get_demo_data():
    """Pre-compute track list and metrics from test predictions."""
    df = load_test_predictions()
    names = load_track_names()

    # Identify tracks with 2+ actual target countries (interesting cases)
    actual_counts = (
        df[df["did_enter_within_60d"] == 1]
        .groupby("track_id")["target_country"]
        .nunique()
        .reset_index(name="actual_countries")
    )
    interesting_tracks = actual_counts[actual_counts["actual_countries"] >= 2]

    # Compute per-track metrics
    track_metrics = []
    for track_id in interesting_tracks["track_id"]:
        track_df = df[df["track_id"] == track_id].copy()
        actuals = set(track_df[track_df["did_enter_within_60d"] == 1]["target_country"])
        top5 = track_df.nsmallest(TOP_K, "predicted_rank")
        top5_countries = set(top5["target_country"])
        hits = len(actuals & top5_countries)
        n_actual = len(actuals)
        track_metrics.append({
            "track_id": track_id,
            "actual_countries": n_actual,
            "top5_hits": hits,
            "recall": hits / n_actual if n_actual else 0,
        })

    track_summary = pd.DataFrame(track_metrics)
    # Join track names
    track_summary = track_summary.merge(names, on="track_id", how="left")
    track_summary["title"] = track_summary["title"].fillna("Unknown")
    track_summary["artist"] = track_summary["artist"].fillna("Unknown")
    track_summary["display_name"] = track_summary["title"] + " — " + track_summary["artist"]
    track_summary = track_summary.sort_values("actual_countries", ascending=False).reset_index(drop=True)

    return df, track_summary


def render():
    st.title("Demo Mode — Test Set Explorer")
    st.markdown("Browse pre-computed predictions from the **2021 test set**. "
                "Showing tracks that spread to 2+ countries.")

    df, track_summary = get_demo_data()

    # ── Metrics header ────────────────────────────────────────────────────
    pe = load_pipeline_evaluation()
    test_row = pe[(pe["split"] == "test") & (pe["gate"] == "none (recommended)")]
    if not test_row.empty:
        r = test_row.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recall@5", f"{r['recall@5']:.3f}")
        col2.metric("Hit Rate@5", f"{r['hit_rate@5']:.3f}")
        col3.metric("NDCG@5", f"{r['ndcg@5']:.3f}")
        col4.metric("Timing MAE", f"{r['days_mae']:.1f} days")

    st.divider()

    # ── Track selector ────────────────────────────────────────────────────
    col_select, col_random = st.columns([3, 1])

    with col_random:
        st.write("")  # spacer
        st.write("")
        if st.button("Random track"):
            random_track = track_summary.sample(1).iloc[0]["track_id"]
            st.session_state["selected_track"] = random_track

    with col_select:
        # Build display options — keyed by track_id, shown as "Title — Artist"
        options = track_summary["track_id"].tolist()
        label_map = {
            row["track_id"]: (
                f"{row['display_name']}  ({row['actual_countries']} countries, "
                f"{row['top5_hits']}/{TOP_K} hits)"
            )
            for row in track_summary.to_dict("records")
        }

        default_idx = 0
        if "selected_track" in st.session_state:
            try:
                default_idx = options.index(st.session_state["selected_track"])
            except ValueError:
                default_idx = 0

        selected = st.selectbox(
            f"Select a track ({len(options)} tracks with 2+ spread countries)",
            options=options,
            format_func=lambda x: label_map.get(x, x),
            index=default_idx,
        )

    if not selected:
        return

    # ── Prediction results for selected track ─────────────────────────────
    track_row = track_summary[track_summary["track_id"] == selected].iloc[0]
    track_df = df[df["track_id"] == selected].copy()
    actuals = set(track_df[track_df["did_enter_within_60d"] == 1]["target_country"])

    # Origin chart footprint — the model input at observation time
    obs_time = track_df["observation_time"].iloc[0]
    origin_footprint = _get_origin_footprint(selected, track_df)
    if not origin_footprint.empty:
        st.markdown(f"**Charting at observation time** ({obs_time.date()}) — model input:")
        st.dataframe(origin_footprint, use_container_width=True, hide_index=True)
    else:
        st.info(f"Not charting in any country at observation time ({obs_time.date()}).")

    # Top-5 predictions
    top5 = track_df.nsmallest(TOP_K, "predicted_rank").copy()
    top5["correct"] = top5["target_country"].isin(actuals)
    top5["actual_days"] = top5.apply(
        lambda r: r["days_to_entry"] if r["did_enter_within_60d"] == 1 else None, axis=1,
    )

    st.subheader(f"{track_row['display_name']} — Top-{TOP_K} Predictions")

    hits = top5["correct"].sum()
    st.markdown(f"**{hits} of {TOP_K}** predictions correct")

    # Display table
    display_df = top5[["predicted_rank", "target_country", "rank_score", "predicted_days_to_entry", "actual_days", "correct"]].copy()
    display_df.columns = ["Rank", "Country", "Score", "Predicted Days", "Actual Days", "Correct"]
    display_df["Correct"] = display_df["Correct"].map({True: "✅", False: "❌"})
    display_df["Score"] = display_df["Score"].round(3)
    display_df["Predicted Days"] = display_df["Predicted Days"].round(1)
    display_df["Actual Days"] = display_df["Actual Days"].round(1)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Missed countries
    missed = actuals - set(top5["target_country"])
    if missed:
        st.markdown(f"**Missed countries** (actual spread targets not in top-{TOP_K}):")
        # Show with their actual days
        missed_df = track_df[track_df["target_country"].isin(missed)][
            ["target_country", "days_to_entry", "predicted_rank", "rank_score"]
        ].sort_values("days_to_entry")
        missed_df.columns = ["Country", "Actual Days", "Predicted Rank", "Score"]
        missed_df["Score"] = missed_df["Score"].round(3)
        missed_df["Actual Days"] = missed_df["Actual Days"].round(1)
        st.dataframe(missed_df, use_container_width=True, hide_index=True)

    # ── Feature importance (expandable) ───────────────────────────────────
    with st.expander("Feature Importance"):
        fi = load_feature_importance()
        fi_sorted = fi.sort_values("gain", ascending=False).head(20)
        st.bar_chart(fi_sorted.set_index("feature")["gain"])

    # ── Model comparison (expandable) ─────────────────────────────────────
    with st.expander("Model Comparison"):
        mc = load_model_comparison()
        keep = ["Naive popularity", "NB12 Final (standalone)"]
        mc = mc[mc["notebook"].isin(keep)].copy()
        mc["notebook"] = mc["notebook"].replace({
            "NB12 Final (standalone)": "XGBRanker Pipeline",
            "Naive popularity": "Naive Popularity Baseline",
        })
        # Add logistic regression baseline from training summary
        from src.config import TRAINING_SUMMARY
        lr = TRAINING_SUMMARY["baseline_logistic_regression"]
        lr_row = pd.DataFrame([{
            "notebook": "Logistic Regression Baseline",
            "recall@5": lr["test_recall"],
            "ndcg@5": lr["test_ndcg"],
            "hit_rate@5": lr["test_hit_rate"],
        }])
        mc = pd.concat([mc, lr_row], ignore_index=True)
        mc = mc.rename(columns={"notebook": "Model"})
        # Sort: baseline → logistic → xgb
        order = ["Naive Popularity Baseline", "Logistic Regression Baseline", "XGBRanker Pipeline"]
        mc["_sort"] = mc["Model"].map({m: i for i, m in enumerate(order)})
        mc = mc.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
        st.dataframe(mc, use_container_width=True, hide_index=True)

        # Show evaluation plot if it exists
        plot_path = EVAL_DIR / "final_evaluation_plots.png"
        if plot_path.exists():
            st.image(str(plot_path), caption="Final Evaluation Plots")
