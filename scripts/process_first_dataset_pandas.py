#!/usr/bin/env python3
import argparse
import csv
import shutil
import sqlite3
import re
from pathlib import Path

import pandas as pd


RAW_NULL_PROFILE_COLUMNS = [
    "title",
    "rank",
    "date",
    "artist",
    "region",
    "chart",
    "streams",
    "track_id",
    "release_date",
]

AUX_OUTPUT_COLUMNS = [
    "source_country_norm",
    "country_continent",
    "country_population",
    "country_area",
    "country_official_language",
    "country_major_religions",
    "country_govt_type",
    "country_driving_side",
    "cultural_distance_mean",
    "cultural_distance_median",
    "cultural_distance_min",
    "cultural_distance_max",
    "cultural_distance_count",
    "cultural_top5_targets",
]

SLIM_PREFERRED_COLUMNS = [
    "observation_date",
    "year",
    "year_month",
    "track_id",
    "title",
    "artist",
    "region",
    "chart",
    "rank",
    "streams",
    "popularity",
    "duration_ms",
    "explicit",
    "release_date",
    "af_danceability",
    "af_energy",
    "af_key",
    "af_loudness",
    "af_mode",
    "af_speechiness",
    "af_acousticness",
    "af_instrumentalness",
    "af_liveness",
    "af_valence",
    "af_tempo",
    "af_time_signature",
    "country_continent",
    "country_population",
    "country_area",
    "cultural_distance_mean",
    "cultural_distance_median",
    "cultural_distance_min",
    "cultural_distance_max",
    "cultural_distance_count",
]


def snake_case(name: str) -> str:
    n = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return n or "unnamed"


def normalize_country_name(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def is_missing(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    return s.isna() | (s.str.strip() == "")


def normalize_raw_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: snake_case(c) for c in df.columns})
    drop_cols = [c for c in df.columns if c == "unnamed" or c.startswith("unnamed_")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def init_distinct_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("CREATE TABLE IF NOT EXISTS distinct_track_id (value TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS distinct_artist (value TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS distinct_region (value TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS distinct_chart (value TEXT PRIMARY KEY)")


def insert_distinct(conn: sqlite3.Connection, table: str, series: pd.Series) -> None:
    cleaned = series.astype("string").str.strip()
    vals = sorted(set(v for v in cleaned.dropna().tolist() if v != ""))
    if vals:
        conn.executemany(
            f"INSERT OR IGNORE INTO {table}(value) VALUES (?)",
            [(v,) for v in vals],
        )


def load_aux_lookup(aux_root: Path) -> pd.DataFrame:
    countries_path = aux_root / "countries_reference_clean.parquet"
    cultural_long_path = aux_root / "cultural_distance_long.parquet"
    cultural_top5_path = aux_root / "cultural_distance_top5.parquet"

    if not countries_path.exists() or not cultural_long_path.exists():
        raise FileNotFoundError(
            f"Aux files missing under {aux_root}. Run scripts/prepare_auxiliary_datasets.sh first."
        )

    countries = pd.read_parquet(countries_path).copy()
    countries["source_country_norm"] = normalize_country_name(countries["country"])
    countries = countries.rename(
        columns={
            "continent": "country_continent",
            "population": "country_population",
            "area": "country_area",
            "official_language": "country_official_language",
            "major_religions": "country_major_religions",
            "govt_type": "country_govt_type",
            "driving_side": "country_driving_side",
        }
    )
    countries = countries[
        [
            "source_country_norm",
            "country_continent",
            "country_population",
            "country_area",
            "country_official_language",
            "country_major_religions",
            "country_govt_type",
            "country_driving_side",
        ]
    ].drop_duplicates(subset=["source_country_norm"], keep="first")

    cultural = pd.read_parquet(cultural_long_path).copy()
    cultural["source_country_norm"] = normalize_country_name(cultural["source_country"])
    cultural["cultural_distance"] = pd.to_numeric(cultural["cultural_distance"], errors="coerce")
    cultural_agg = (
        cultural.groupby("source_country_norm", as_index=False)["cultural_distance"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "cultural_distance_mean",
                "median": "cultural_distance_median",
                "min": "cultural_distance_min",
                "max": "cultural_distance_max",
                "count": "cultural_distance_count",
            }
        )
    )

    if cultural_top5_path.exists():
        top5 = pd.read_parquet(cultural_top5_path).copy()
        top5["source_country_norm"] = normalize_country_name(top5["source_country"])
        top5_targets = (
            top5.sort_values(["source_country_norm", "rank_within_source"])
            .groupby("source_country_norm", as_index=False)["target_country"]
            .agg(lambda x: "|".join(x.astype(str).tolist()))
            .rename(columns={"target_country": "cultural_top5_targets"})
        )
    else:
        top5_targets = pd.DataFrame(columns=["source_country_norm", "cultural_top5_targets"])

    aux = countries.merge(cultural_agg, on="source_country_norm", how="left")
    aux = aux.merge(top5_targets, on="source_country_norm", how="left")

    aux["country_population"] = pd.to_numeric(aux["country_population"], errors="coerce").astype("Int64")
    aux["country_area"] = pd.to_numeric(aux["country_area"], errors="coerce").astype("Float32")
    for c in ["cultural_distance_mean", "cultural_distance_median", "cultural_distance_min", "cultural_distance_max"]:
        aux[c] = pd.to_numeric(aux[c], errors="coerce").astype("Float32")
    aux["cultural_distance_count"] = pd.to_numeric(aux["cultural_distance_count"], errors="coerce").astype("Int64")

    return aux


def add_merge_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" not in out.columns:
        out["date"] = pd.NA

    observation_dt = pd.to_datetime(out["date"], errors="coerce", format="%Y-%m-%d")
    out["observation_date"] = observation_dt.dt.strftime("%Y-%m-%d")
    out.loc[observation_dt.isna(), "observation_date"] = pd.NA

    out["year_month"] = out["observation_date"].astype("string").str.slice(0, 7)
    out["year_month"] = out["year_month"].fillna("unknown")
    out["year"] = out["observation_date"].astype("string").str.slice(0, 4)
    out["year"] = out["year"].fillna("unknown")

    if "region" not in out.columns:
        out["region"] = pd.NA
    out["source_country_norm"] = normalize_country_name(out["region"])

    return out


def apply_aux_features(df: pd.DataFrame, aux_lookup: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()

    for col in AUX_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    if aux_lookup is None:
        return out

    merged = out.merge(aux_lookup, on="source_country_norm", how="left", suffixes=("", "_aux"))
    for col in AUX_OUTPUT_COLUMNS:
        aux_col = f"{col}_aux"
        if aux_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[aux_col])
            merged = merged.drop(columns=[aux_col])
    return merged


def clean_output_root(output_root: Path) -> None:
    for sub in ["full", "slim"]:
        p = output_root / sub
        if p.exists():
            shutil.rmtree(p)

    for file_name in ["source_schema.csv", "profile_overview.csv", "profile_null_rates.csv", "row_accounting.csv", "quarantine.parquet"]:
        f = output_root / file_name
        if f.exists():
            f.unlink()


def write_partitioned_chunk(df: pd.DataFrame, root: Path, counters: dict[str, int]) -> None:
    if df.empty:
        return

    for year, part in df.groupby("year", dropna=False):
        year_str = str(year) if pd.notna(year) and str(year).strip() != "" else "unknown"
        year_str = year_str.replace("/", "-")
        part_dir = root / f"year={year_str}"
        part_dir.mkdir(parents=True, exist_ok=True)
        idx = counters.get(year_str, 0)
        part.to_parquet(part_dir / f"part-{idx:05d}.parquet", index=False, compression="zstd")
        counters[year_str] = idx + 1


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: merge-only dataset processing using pandas chunks.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--db-file", required=True)
    parser.add_argument("--chunksize", type=int, default=150_000)
    parser.add_argument("--join-aux", action="store_true", help="Join prepared auxiliary country/cultural features")
    parser.add_argument("--aux-root", default="datasets/processed/v1_aux", help="Path containing prepared aux parquet tables")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_root = Path(args.output_root)
    db_file = Path(args.db_file)

    aux_lookup = None
    if args.join_aux:
        aux_lookup = load_aux_lookup(Path(args.aux_root))

    output_root.mkdir(parents=True, exist_ok=True)
    clean_output_root(output_root)

    conn = sqlite3.connect(db_file)
    init_distinct_db(conn)

    raw_rows = 0
    merged_rows = 0
    min_date = None
    max_date = None
    source_schema_rows = []
    null_counts = {c: 0 for c in RAW_NULL_PROFILE_COLUMNS}

    full_counters: dict[str, int] = {}
    slim_counters: dict[str, int] = {}

    reader = pd.read_csv(input_csv, chunksize=args.chunksize, dtype="string", low_memory=False)

    full_columns: list[str] | None = None

    for chunk_index, raw_chunk in enumerate(reader):
        raw_chunk = normalize_raw_chunk(raw_chunk)

        if chunk_index == 0:
            source_schema_rows = [[i, c, "STRING", 0, "", 0] for i, c in enumerate(raw_chunk.columns.tolist())]

        chunk_rows = len(raw_chunk)
        raw_rows += chunk_rows

        if "date" in raw_chunk.columns:
            dt = pd.to_datetime(raw_chunk["date"], errors="coerce", format="%Y-%m-%d")
            chunk_min = dt.min()
            chunk_max = dt.max()
            if pd.notna(chunk_min):
                min_date = chunk_min if min_date is None else min(min_date, chunk_min)
            if pd.notna(chunk_max):
                max_date = chunk_max if max_date is None else max(max_date, chunk_max)

        for col in RAW_NULL_PROFILE_COLUMNS:
            if col in raw_chunk.columns:
                null_counts[col] += int(is_missing(raw_chunk[col]).sum())
            else:
                null_counts[col] += chunk_rows

        if "track_id" in raw_chunk.columns:
            insert_distinct(conn, "distinct_track_id", raw_chunk["track_id"])
        if "artist" in raw_chunk.columns:
            insert_distinct(conn, "distinct_artist", raw_chunk["artist"])
        if "region" in raw_chunk.columns:
            insert_distinct(conn, "distinct_region", raw_chunk["region"])
        if "chart" in raw_chunk.columns:
            insert_distinct(conn, "distinct_chart", raw_chunk["chart"])

        merged = add_merge_columns(raw_chunk)
        merged = apply_aux_features(merged, aux_lookup)

        if full_columns is None:
            full_columns = list(merged.columns)
        else:
            missing_cols = [c for c in full_columns if c not in merged.columns]
            for c in missing_cols:
                merged[c] = pd.NA
            extra_cols = [c for c in merged.columns if c not in full_columns]
            if extra_cols:
                full_columns.extend(extra_cols)
            merged = merged[full_columns]

        merged_rows += len(merged)

        full_dir = output_root / "full"
        write_partitioned_chunk(merged, full_dir, full_counters)

        slim_cols = [c for c in SLIM_PREFERRED_COLUMNS if c in merged.columns]
        slim = merged[slim_cols].copy()
        slim_dir = output_root / "slim"
        write_partitioned_chunk(slim, slim_dir, slim_counters)

    conn.commit()

    profile_overview_rows = [[
        raw_rows,
        merged_rows,
        min_date.strftime("%Y-%m-%d") if min_date is not None else "",
        max_date.strftime("%Y-%m-%d") if max_date is not None else "",
        conn.execute("SELECT COUNT(*) FROM distinct_track_id").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM distinct_artist").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM distinct_region").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM distinct_chart").fetchone()[0],
        int(args.join_aux),
    ]]

    null_rate_rows = [[c, (null_counts[c] / raw_rows if raw_rows else 0.0)] for c in RAW_NULL_PROFILE_COLUMNS]

    row_accounting_rows = [
        ["raw_rows", raw_rows],
        ["merged_rows", merged_rows],
        ["rows_filtered", 0],
    ]

    write_csv(output_root / "source_schema.csv", ["cid", "name", "type", "not_null", "dflt_value", "pk"], source_schema_rows)
    write_csv(
        output_root / "profile_overview.csv",
        [
            "raw_rows",
            "merged_rows",
            "min_date",
            "max_date",
            "distinct_track_id",
            "distinct_artist",
            "distinct_region",
            "distinct_chart",
            "join_aux_enabled",
        ],
        profile_overview_rows,
    )
    write_csv(output_root / "profile_null_rates.csv", ["column_name", "null_rate"], null_rate_rows)
    write_csv(output_root / "row_accounting.csv", ["metric", "value"], row_accounting_rows)

    conn.close()


if __name__ == "__main__":
    main()
