#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def normalize_country_name(series: pd.Series) -> pd.Series:
    s = series.astype('string').str.strip()
    return s.str.replace(r'\s+', ' ', regex=True)


def transform_cultural_matrix(input_path: Path, output_root: Path) -> dict:
    df = pd.read_csv(input_path)
    df = df.rename(columns=lambda c: str(c).strip())
    if len(df.columns) < 2:
        raise ValueError('cultural matrix must contain one source-country column and at least one target-country column')

    source_column_candidates = {
        'country',
        'source_country',
        'source',
        'country_name',
        'countries',
    }
    source_column = next((c for c in df.columns if str(c).strip().lower() in source_column_candidates), None)
    if source_column is None:
        source_column = df.columns[0]

    df[source_column] = normalize_country_name(df[source_column])

    long_df = df.melt(
        id_vars=[source_column],
        var_name='target_country',
        value_name='cultural_distance',
    ).rename(columns={source_column: 'source_country'})

    long_df['target_country'] = normalize_country_name(long_df['target_country'])
    long_df['cultural_distance'] = pd.to_numeric(long_df['cultural_distance'], errors='coerce')

    long_df = long_df.dropna(subset=['source_country', 'target_country', 'cultural_distance']).copy()
    long_df = long_df[long_df['source_country'] != long_df['target_country']].copy()

    # Keep one record per directed pair.
    long_df = long_df.sort_values(
        ['source_country', 'target_country', 'cultural_distance'],
        ascending=[True, True, True],
    ).drop_duplicates(subset=['source_country', 'target_country'], keep='first')

    long_df['cultural_distance'] = long_df['cultural_distance'].astype('float32')

    # Optional helper table: 5 nearest target countries per source.
    nearest5 = (
        long_df.sort_values(['source_country', 'cultural_distance', 'target_country'])
        .groupby('source_country', as_index=False)
        .head(5)
        .copy()
    )
    nearest5['rank_within_source'] = nearest5.groupby('source_country').cumcount() + 1
    nearest5['rank_within_source'] = nearest5['rank_within_source'].astype('int16')

    output_root.mkdir(parents=True, exist_ok=True)
    long_path = output_root / 'cultural_distance_long.parquet'
    nearest_path = output_root / 'cultural_distance_top5.parquet'

    long_df.to_parquet(long_path, index=False, compression='zstd')
    nearest5.to_parquet(nearest_path, index=False, compression='zstd')

    return {
        'rows_long': int(len(long_df)),
        'rows_top5': int(len(nearest5)),
        'path_long': str(long_path),
        'path_top5': str(nearest_path),
    }


def transform_country_metadata(input_path: Path, output_root: Path) -> dict:
    df = pd.read_csv(input_path)
    df = df.rename(columns=lambda c: str(c).strip())

    rename_map = {
        'country_name': 'country',
        'official_lang': 'official_language',
        'Major Religions': 'major_religions',
    }
    df = df.rename(columns=rename_map)

    keep_cols = [
        'country',
        'continent',
        'population',
        'area',
        'official_language',
        'major_religions',
        'govt_type',
        'driving_side',
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA

    out = df[keep_cols].copy()
    out['country'] = normalize_country_name(out['country'])
    out['population'] = pd.to_numeric(out['population'], errors='coerce').astype('Int64')
    out['area'] = pd.to_numeric(out['area'], errors='coerce').astype('Float32')

    # Lightweight normalization for text columns.
    for col in ['continent', 'official_language', 'major_religions', 'govt_type', 'driving_side']:
        out[col] = out[col].astype('string').str.strip().replace('', pd.NA)

    out = out.dropna(subset=['country']).drop_duplicates(subset=['country'], keep='first')

    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / 'countries_reference_clean.parquet'
    out.to_parquet(out_path, index=False, compression='zstd')

    return {
        'rows': int(len(out)),
        'path': str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare auxiliary datasets for joins.')
    parser.add_argument('--countries-csv', default='datasets/Countries Data By Aadarsh Vani.csv')
    parser.add_argument('--cultural-matrix-csv', default='datasets/cultural_distance_matrix.csv')
    parser.add_argument('--output-root', default='datasets/processed/v1_aux')
    args = parser.parse_args()

    output_root = Path(args.output_root)

    cultural_stats = transform_cultural_matrix(Path(args.cultural_matrix_csv), output_root)
    country_stats = transform_country_metadata(Path(args.countries_csv), output_root)

    report = pd.DataFrame([
        {'artifact': 'cultural_distance_long', 'rows': cultural_stats['rows_long'], 'path': cultural_stats['path_long']},
        {'artifact': 'cultural_distance_top5', 'rows': cultural_stats['rows_top5'], 'path': cultural_stats['path_top5']},
        {'artifact': 'countries_reference_clean', 'rows': country_stats['rows'], 'path': country_stats['path']},
    ])
    report.to_csv(output_root / 'aux_profile.csv', index=False)

    print('Auxiliary dataset processing finished.')
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
