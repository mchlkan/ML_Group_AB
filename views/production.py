"""Production Mode: custom song input → top-5 country predictions."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.config import (
    COUNTRY_LIST,
    COUNTRY_PRIMARY_LANG,
    FILL_VALUES_FINAL,
    PRUNED_ROW_FEATURE_COLS,
    SONG_LANGUAGES,
    TOP_K,
)
from src.data import build_prediction_rows, load_reference_data, lookup_artist
from src.models import load_pretrained_models, predict_custom_song


def render():
    st.title("Production Mode — Custom Song Prediction")
    st.markdown(
        "Enter song details below. Features that can be derived automatically "
        "(artist history, cultural distances, etc.) are computed for you."
    )

    # Load models and reference data upfront
    models = load_pretrained_models()
    reference_data = load_reference_data()

    # ── Input form ────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Song Info")
        artist_name = st.text_input("Artist name", placeholder="e.g. Bad Bunny")
        song_title = st.text_input("Song title", placeholder="e.g. Dakiti")

        lang_options = [code for code, _ in SONG_LANGUAGES]
        lang_labels = [f"{label} ({code})" for code, label in SONG_LANGUAGES]
        song_language = st.selectbox(
            "Song language", options=lang_options,
            format_func=lambda x: lang_labels[lang_options.index(x)],
        )

        release_date = st.date_input("Release date", value=date.today())
        explicit = st.checkbox("Explicit")
        on_viral50 = st.checkbox("On Viral 50?")

    with col_right:
        st.subheader("Chart Footprint")
        st.caption("Add countries where the song is currently charting.")

        if "chart_footprint" not in st.session_state:
            st.session_state.chart_footprint = []

        fp_col1, fp_col2, fp_col3 = st.columns([2, 1, 1])
        with fp_col1:
            fp_country = st.selectbox(
                "Country", options=COUNTRY_LIST, key="fp_country",
            )
        with fp_col2:
            fp_rank = st.slider("Rank", 1, 200, 50, key="fp_rank")
        with fp_col3:
            st.write("")  # spacer
            st.write("")
            if st.button("Add"):
                # Remove existing entry for this country if any
                st.session_state.chart_footprint = [
                    e for e in st.session_state.chart_footprint
                    if e["country"] != fp_country
                ]
                st.session_state.chart_footprint.append(
                    {"country": fp_country, "rank": fp_rank}
                )

        # Show current footprint
        if st.session_state.chart_footprint:
            fp_df = pd.DataFrame(st.session_state.chart_footprint)
            st.dataframe(fp_df, use_container_width=True, hide_index=True)
            if st.button("Clear footprint"):
                st.session_state.chart_footprint = []
                st.rerun()
        else:
            st.info("No chart footprint added yet. The song will be treated as new.")

    # ── Audio features (collapsible) ──────────────────────────────────────
    with st.expander("Audio Features (pre-filled with training medians)"):
        st.caption("These barely affect predictions — adjust only if you have the actual values.")
        ac1, ac2, ac3 = st.columns(3)
        audio_features = {}
        with ac1:
            audio_features["af_danceability"] = st.slider(
                "Danceability", 0.0, 1.0, float(FILL_VALUES_FINAL["af_danceability"]), 0.01,
            )
            audio_features["af_energy"] = st.slider(
                "Energy", 0.0, 1.0, float(FILL_VALUES_FINAL["af_energy"]), 0.01,
            )
            audio_features["af_valence"] = st.slider(
                "Valence", 0.0, 1.0, float(FILL_VALUES_FINAL["af_valence"]), 0.01,
            )
            audio_features["af_acousticness"] = st.slider(
                "Acousticness", 0.0, 1.0, float(FILL_VALUES_FINAL["af_acousticness"]), 0.01,
            )
        with ac2:
            audio_features["af_speechiness"] = st.slider(
                "Speechiness", 0.0, 1.0, float(FILL_VALUES_FINAL["af_speechiness"]), 0.01,
            )
            audio_features["af_instrumentalness"] = st.slider(
                "Instrumentalness", 0.0, 1.0, float(FILL_VALUES_FINAL["af_instrumentalness"]), 0.01,
            )
            audio_features["af_liveness"] = st.slider(
                "Liveness", 0.0, 1.0, float(FILL_VALUES_FINAL["af_liveness"]), 0.01,
            )
            audio_features["af_loudness"] = st.slider(
                "Loudness (dB)", -20.0, 0.0, float(FILL_VALUES_FINAL["af_loudness"]), 0.1,
            )
        with ac3:
            audio_features["af_tempo"] = st.slider(
                "Tempo (BPM)", 50.0, 220.0, float(FILL_VALUES_FINAL["af_tempo"]), 1.0,
            )
            audio_features["duration_ms"] = st.slider(
                "Duration (ms)", 60000, 600000, int(FILL_VALUES_FINAL["duration_ms"]), 1000,
            )
            audio_features["af_key"] = st.slider(
                "Key (0-11)", 0, 11, int(FILL_VALUES_FINAL["af_key"]),
            )
            audio_features["af_mode"] = st.selectbox(
                "Mode", [0, 1], index=int(FILL_VALUES_FINAL["af_mode"]),
                format_func=lambda x: "Major" if x == 1 else "Minor",
            )
            audio_features["af_time_signature"] = st.selectbox(
                "Time Signature", [3, 4, 5], index=1,
            )

    # ── Predict button ────────────────────────────────────────────────────
    st.divider()

    if st.button("Predict", type="primary", use_container_width=True):
        if not artist_name.strip():
            st.warning("Please enter an artist name.")
            return

        with st.spinner("Looking up artist history..."):
            artist_info = lookup_artist(artist_name)

        # Show artist lookup results
        if artist_info["artist_prior_chart_count"] > 0:
            st.success(
                f"Found **{artist_info['artist_prior_unique_tracks']}** tracks by "
                f"**{artist_name}** across **{artist_info['artist_prior_unique_regions']}** "
                f"regions (best rank: #{artist_info['artist_prior_best_rank']})"
            )
        else:
            st.info(f"**{artist_name}** not found in dataset — using zero-history defaults.")

        song_input = {
            "artist_name": artist_name,
            "song_title": song_title,
            "song_language": song_language,
            "chart_footprint": st.session_state.chart_footprint,
            "on_viral50": on_viral50,
            "release_date": release_date,
            "explicit": explicit,
            "audio_features": audio_features,
        }

        with st.spinner("Building prediction rows..."):
            prediction_df = build_prediction_rows(song_input, reference_data, artist_info)

        with st.spinner("Running model prediction..."):
            origin_countries = {e["country"] for e in st.session_state.chart_footprint}
            results = predict_custom_song(
                models, prediction_df, PRUNED_ROW_FEATURE_COLS, FILL_VALUES_FINAL,
                origin_countries=origin_countries,
            )

        st.divider()
        _display_results(results, song_input, artist_info)


def _display_results(results: dict, song_input: dict, artist_info: dict):
    """Display prediction results."""
    top_k = results["top_k"]
    all_scores = results["all_scores"]

    st.subheader(f"Top-{TOP_K} Predicted Countries")
    st.caption(f"Prediction took {results['timing_ms']:.0f} ms")

    # Top-5 table
    display_df = top_k[["predicted_rank", "target_country", "score", "predicted_days_to_entry"]].copy()
    display_df.columns = ["Rank", "Country", "Score", "Predicted Days to Entry"]
    display_df["Score"] = display_df["Score"].round(3)
    display_df["Predicted Days to Entry"] = display_df["Predicted Days to Entry"].round(1)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Bar chart of all 62 country scores
    st.subheader("All Country Scores")
    chart_df = all_scores[["target_country", "score"]].copy()
    chart_df = chart_df.set_index("target_country").sort_values("score", ascending=True)
    st.bar_chart(chart_df, horizontal=True, height=900)

    # Explanation section
    with st.expander("Why these countries?"):
        song_lang = song_input.get("song_language", "en")
        origin_countries = {e["country"] for e in song_input.get("chart_footprint", [])}

        explanations = []
        for _, row in top_k.iterrows():
            country = row["target_country"]
            reasons = []
            target_lang = COUNTRY_PRIMARY_LANG.get(country, "")
            if song_lang == target_lang:
                reasons.append("song language matches")
            if country in artist_info.get("charted_countries", set()):
                reasons.append("artist has charted here before")
            for oc in origin_countries:
                if COUNTRY_PRIMARY_LANG.get(oc, "") == target_lang:
                    reasons.append(f"shares language with {oc}")
                    break
            if not reasons:
                reasons.append("model-inferred cultural/market proximity")
            explanations.append(f"**{country}**: {', '.join(reasons)}")

        st.markdown("\n\n".join(explanations))
