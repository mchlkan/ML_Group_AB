"""Data loading (DuckDB), artist lookup, reference data, and prediction row construction."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    COUNTRY_CONTINENT,
    COUNTRY_LIST,
    COUNTRY_PRIMARY_LANG,
    CONTINENT_ONEHOT_COLS,
    CULTURAL_DIST_CSV,
    COUNTRIES_CSV,
    FILL_VALUES_FINAL,
    PRUNED_ROW_FEATURE_COLS,
    V2_DATA_DIR,
    country_to_rank_col,
)


# ---------------------------------------------------------------------------
# Data loading (from notebook)
# ---------------------------------------------------------------------------

FEATURE_EXCLUDE = [
    "track_id", "observation_time", "target_country",
    "did_enter_within_60d", "days_to_entry",
]


def load_row_level_split(path, max_tracks: int | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    parquet_path = path.as_posix()
    if max_tracks is None:
        query = f"SELECT * FROM read_parquet('{parquet_path}')"
    else:
        query = f"""
            WITH sampled_tracks AS (
                SELECT track_id
                FROM read_parquet('{parquet_path}')
                GROUP BY track_id
                ORDER BY hash(track_id)
                LIMIT {int(max_tracks)}
            )
            SELECT d.*
            FROM read_parquet('{parquet_path}') d
            JOIN sampled_tracks st USING (track_id)
        """
    df = con.execute(query).fetchdf()
    df["observation_time"] = pd.to_datetime(df["observation_time"])
    con.close()
    return df


def make_feature_matrix(
    df: pd.DataFrame, feature_cols: list[str], fill_values: dict | pd.Series,
) -> pd.DataFrame:
    return df[feature_cols].copy().fillna(fill_values)


def prepare_ranker_inputs(
    df: pd.DataFrame, feature_cols: list[str], fill_values: dict | pd.Series,
):
    ordered = df.sort_values(["track_id", "target_country"]).reset_index(drop=True)
    X = make_feature_matrix(ordered, feature_cols, fill_values)
    y = ordered["did_enter_within_60d"].astype(float).to_numpy()
    group = ordered.groupby("track_id", sort=False).size().to_numpy()
    return ordered, X, y, group


# ---------------------------------------------------------------------------
# Artist lookup (NEW)
# ---------------------------------------------------------------------------

_V2_PARQUET = f"read_parquet('{V2_DATA_DIR.as_posix()}/*/*.parquet')"


def lookup_artist(artist_name: str) -> dict:
    """Query v2 dataset for an artist's prior chart history.
    Returns dict with artist_prior_chart_count, artist_prior_unique_regions,
    artist_prior_best_rank, artist_prior_unique_tracks, multi_artist_flag.
    Returns zeros if artist not found.
    """
    zeros = {
        "artist_prior_chart_count": 0,
        "artist_prior_unique_regions": 0,
        "artist_prior_best_rank": 200,
        "artist_prior_unique_tracks": 0,
        "multi_artist_flag": 0,
        "artist_country_ratio": 0.0,
        "charted_countries": set(),
    }
    if not artist_name or not artist_name.strip():
        return zeros

    con = duckdb.connect()
    try:
        result = con.execute(
            f"""
            SELECT
                COUNT(*) as chart_count,
                COUNT(DISTINCT source_country_norm) as unique_regions,
                MIN(rank) as best_rank,
                COUNT(DISTINCT track_id) as unique_tracks,
                LIST(DISTINCT source_country_norm) as countries
            FROM {_V2_PARQUET}
            WHERE LOWER(artist) LIKE '%' || LOWER(?) || '%'
            """,
            [artist_name.strip()],
        ).fetchdf()

        if result.empty or result.iloc[0]["chart_count"] == 0:
            return zeros

        row = result.iloc[0]
        raw_countries = row["countries"]
        countries = set(raw_countries) if raw_countries is not None and len(raw_countries) > 0 else set()
        return {
            "artist_prior_chart_count": int(row["chart_count"]),
            "artist_prior_unique_regions": int(row["unique_regions"]),
            "artist_prior_best_rank": int(row["best_rank"]),
            "artist_prior_unique_tracks": int(row["unique_tracks"]),
            "multi_artist_flag": 0,
            "artist_country_ratio": (
                int(row["unique_regions"]) / 62.0 if row["unique_regions"] else 0.0
            ),
            "charted_countries": countries,
        }
    except Exception as e:
        st.warning(f"Artist lookup failed: {e}")
        return zeros
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Reference data (NEW)
# ---------------------------------------------------------------------------

@st.cache_data
def load_reference_data() -> dict:
    """Load country metadata and cultural distance matrix. Returns dict with
    'countries_df', 'cultural_dist_df', and 'country_metadata'.
    """
    countries_df = pd.read_csv(COUNTRIES_CSV)
    cultural_dist_df = pd.read_csv(CULTURAL_DIST_CSV, index_col="country")

    # Build per-country metadata for the 62 Spotify markets
    country_metadata = {}
    for country in COUNTRY_LIST:
        row = countries_df[countries_df["country_name"] == country]
        population = int(row["population"].iloc[0]) if not row.empty else FILL_VALUES_FINAL.get("target_population", 10_000_000)
        country_metadata[country] = {
            "population": population,
            "continent": COUNTRY_CONTINENT.get(country, ""),
            "primary_lang": COUNTRY_PRIMARY_LANG.get(country, ""),
        }

    return {
        "countries_df": countries_df,
        "cultural_dist_df": cultural_dist_df,
        "country_metadata": country_metadata,
    }


def _get_cultural_distance(cultural_dist_df: pd.DataFrame, origin: str, target: str) -> float | None:
    """Get Hofstede cultural distance between two countries. Returns None if not available."""
    if origin in cultural_dist_df.index and target in cultural_dist_df.columns:
        val = cultural_dist_df.loc[origin, target]
        if pd.notna(val):
            return float(val)
    return None


# ---------------------------------------------------------------------------
# Build prediction rows for custom song (NEW)
# ---------------------------------------------------------------------------

def build_prediction_rows(
    song_input: dict,
    reference_data: dict,
    artist_info: dict,
) -> pd.DataFrame:
    """Create a 62-row DataFrame (one per target country) with all features computed.

    song_input keys:
        artist_name, song_title, song_language, chart_footprint (list of {country, rank}),
        on_viral50, release_date, explicit, audio_features (dict)
    """
    cultural_dist_df = reference_data["cultural_dist_df"]
    country_metadata = reference_data["country_metadata"]

    # Origin countries from chart footprint
    chart_footprint = song_input.get("chart_footprint", [])
    origin_countries = {entry["country"] for entry in chart_footprint}
    origin_ranks = {entry["country"]: entry["rank"] for entry in chart_footprint}

    # Compute days_since_release
    from datetime import date
    release_date = song_input.get("release_date", date.today())
    days_since_release = (date.today() - release_date).days
    is_friday_release = 1 if release_date.weekday() == 4 else 0

    rows = []
    for target in COUNTRY_LIST:
        row = {}

        # Rank columns: set the rank for countries in footprint, 0 otherwise
        for country in COUNTRY_LIST:
            rank_col = country_to_rank_col(country)
            if rank_col in PRUNED_ROW_FEATURE_COLS:
                row[rank_col] = origin_ranks.get(country, 0)

        # Audio features (use provided or defaults from fill values)
        audio_defaults = {
            "af_danceability": FILL_VALUES_FINAL.get("af_danceability", 0.7),
            "af_energy": FILL_VALUES_FINAL.get("af_energy", 0.64),
            "af_valence": FILL_VALUES_FINAL.get("af_valence", 0.5),
            "af_tempo": FILL_VALUES_FINAL.get("af_tempo", 120.0),
            "af_acousticness": FILL_VALUES_FINAL.get("af_acousticness", 0.2),
            "af_speechiness": FILL_VALUES_FINAL.get("af_speechiness", 0.08),
            "af_instrumentalness": FILL_VALUES_FINAL.get("af_instrumentalness", 0.0),
            "af_liveness": FILL_VALUES_FINAL.get("af_liveness", 0.12),
            "af_key": FILL_VALUES_FINAL.get("af_key", 5.0),
            "af_loudness": FILL_VALUES_FINAL.get("af_loudness", -6.8),
            "af_mode": FILL_VALUES_FINAL.get("af_mode", 1.0),
            "af_time_signature": FILL_VALUES_FINAL.get("af_time_signature", 4.0),
            "duration_ms": FILL_VALUES_FINAL.get("duration_ms", 193846.0),
        }
        user_audio = song_input.get("audio_features", {})
        for k, default in audio_defaults.items():
            row[k] = user_audio.get(k, default)

        # Track metadata
        row["explicit"] = 1 if song_input.get("explicit", False) else 0
        row["days_since_release"] = max(days_since_release, 0)
        row["is_friday_release"] = is_friday_release
        row["track_in_viral50_at_obs"] = 1 if song_input.get("on_viral50", False) else 0

        # Artist history
        row["artist_prior_chart_count"] = artist_info["artist_prior_chart_count"]
        row["artist_prior_unique_regions"] = artist_info["artist_prior_unique_regions"]
        row["artist_prior_best_rank"] = artist_info["artist_prior_best_rank"]
        row["artist_prior_unique_tracks"] = artist_info["artist_prior_unique_tracks"]
        row["multi_artist_flag"] = artist_info["multi_artist_flag"]
        row["artist_country_ratio"] = artist_info["artist_country_ratio"]

        # Artist prior success in target
        charted_countries = artist_info.get("charted_countries", set())
        row["artist_prior_success_in_target"] = 1 if target in charted_countries else 0

        # Target country priors
        meta = country_metadata.get(target, {})
        row["target_population"] = meta.get("population", FILL_VALUES_FINAL.get("target_population", 10_000_000))
        row["target_avg_daily_streams"] = FILL_VALUES_FINAL.get("target_avg_daily_streams", 10738.0)
        row["target_new_entry_rate_30d"] = FILL_VALUES_FINAL.get("target_new_entry_rate_30d", 0.051)

        # Continent one-hot
        target_continent = COUNTRY_CONTINENT.get(target, "")
        for col in CONTINENT_ONEHOT_COLS:
            continent_name = col.replace("target_continent_", "").replace("_", " ").title()
            row[col] = 1 if continent_name == target_continent else 0

        # Origin-target relationship features
        song_lang = song_input.get("song_language", "en")
        target_lang = COUNTRY_PRIMARY_LANG.get(target, "")

        # same_language_flag: any origin country shares language with target
        row["same_language_flag"] = 0
        for oc in origin_countries:
            if COUNTRY_PRIMARY_LANG.get(oc, "") == target_lang:
                row["same_language_flag"] = 1
                break

        # song_lang_matches_target
        row["song_lang_matches_target"] = 1 if song_lang == target_lang else 0

        # same_continent_flag
        target_cont = COUNTRY_CONTINENT.get(target, "")
        row["same_continent_flag"] = 0
        for oc in origin_countries:
            if COUNTRY_CONTINENT.get(oc, "") == target_cont:
                row["same_continent_flag"] = 1
                break

        # Cultural distance
        min_dist = None
        for oc in origin_countries:
            d = _get_cultural_distance(cultural_dist_df, oc, target)
            if d is not None:
                min_dist = d if min_dist is None else min(min_dist, d)
        if min_dist is not None:
            row["cultural_dist_min"] = min_dist
            row["cultural_dist_missing"] = 0
        else:
            row["cultural_dist_min"] = FILL_VALUES_FINAL.get("cultural_dist_min", 1.67)
            row["cultural_dist_missing"] = 1

        # neighbor_entered_count: count of target's cultural neighbors in origin set
        row["neighbor_entered_count"] = 0
        if target in cultural_dist_df.index:
            target_dists = cultural_dist_df.loc[target].dropna().sort_values()
            # Top-5 closest countries (excluding self)
            neighbors = [c for c in target_dists.index[:6] if c != target][:5]
            row["neighbor_entered_count"] = len(set(neighbors) & origin_countries)

        # Temporal
        from datetime import date as _date
        today = _date.today()
        row["observation_month"] = today.month
        row["observation_year"] = today.year

        # Metadata for display (not features)
        row["target_country"] = target

        rows.append(row)

    df = pd.DataFrame(rows)
    return df
