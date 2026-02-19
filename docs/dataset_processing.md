# Dataset Processing (Stage 1: Merge Only)

Stage 1 performs **merge-only processing** for teammate EDA.
It does **not** clean, deduplicate, quarantine, or filter rows.

## Goal

Create one merged, analysis-ready dataset from:

- base Spotify dataset (`datasets/merged_data.csv`)
- optional auxiliary tables (countries metadata + cultural distance features)

## Outputs

- `datasets/processed/v1/full/` (partitioned Parquet by `year=YYYY`, zstd)
- `datasets/processed/v1/slim/` (partitioned Parquet by `year=YYYY`, zstd)
- `datasets/processed/v1/source_schema.csv`
- `datasets/processed/v1/profile_overview.csv`
- `datasets/processed/v1/profile_null_rates.csv`
- `datasets/processed/v1/row_accounting.csv`
- `datasets/manifest.v1.json` (or `datasets/manifest.<DATASET_VERSION>.json`)

`row_accounting.csv` for Stage 1 should satisfy:

- `raw_rows = merged_rows`
- `rows_filtered = 0`

## Requirements

- `python3`
- `pandas`
- `pyarrow`

## Run (base merge only)

```bash
bash scripts/process_first_dataset.sh
```

## Run with auxiliary joins

Prepare auxiliary tables first:

```bash
bash scripts/prepare_auxiliary_datasets.sh
```

Then run Stage 1 merge with aux features:

```bash
JOIN_AUX=1 bash scripts/process_first_dataset.sh
```

Optional custom aux root:

```bash
JOIN_AUX=1 AUX_ROOT=/absolute/path/to/v1_aux bash scripts/process_first_dataset.sh
```

Optional custom paths:

```bash
bash scripts/process_first_dataset.sh <input_csv> <output_root> <sqlite_db_file>
```

Optional dataset version (affects default output, aux root, and manifest name):

```bash
DATASET_VERSION=v2 bash scripts/process_first_dataset.sh
```

Optional chunk size:

```bash
CHUNKSIZE=150000 bash scripts/process_first_dataset.sh
```

Chunk size guidance for MacBook Air M4 (16GB):

- recommended default: `150000`
- typical upper practical target: `250000`
- test carefully up to `300000` if stable

## Auxiliary datasets

To prepare join-ready auxiliary tables:

```bash
bash scripts/prepare_auxiliary_datasets.sh
```

Outputs under `datasets/processed/v1_aux/`:

- `cultural_distance_long.parquet`
  - columns: `source_country`, `target_country`, `cultural_distance`
- `cultural_distance_top5.parquet`
  - top-5 nearest target countries per source country
- `countries_reference_clean.parquet`
  - cleaned country metadata
- `aux_profile.csv`
  - artifact row counts and paths

## What Stage 1 merges from auxiliary data

When `JOIN_AUX=1`, the pipeline adds:

- country metadata columns:
  - `country_continent`, `country_population`, `country_area`
  - `country_official_language`, `country_major_religions`, `country_govt_type`, `country_driving_side`
- cultural columns:
  - `cultural_distance_mean`, `cultural_distance_median`, `cultural_distance_min`, `cultural_distance_max`, `cultural_distance_count`
  - `cultural_top5_targets`

Join key:

- base `region` (normalized) -> aux `source_country_norm`

## Next stage

Cleaning is intentionally deferred.
Do EDA on Stage 1 outputs first, then run a separate Stage 2 cleaning pipeline.

## Share dataset via Cloudflare R2

Use the provided helper scripts:

- `scripts/upload_to_r2.sh`
- `scripts/download_from_r2.sh`
- `scripts/r2.env.example`

Setup once:

```bash
cp scripts/r2.env.example scripts/r2.env
# edit scripts/r2.env with your real values
```

The R2 scripts auto-load and export variables from `scripts/r2.env`.

Upload current processed dataset + manifest:

```bash
bash scripts/upload_to_r2.sh
```

Download for teammates:

```bash
bash scripts/download_from_r2.sh
```

Optional overrides:

- Upload script:
  - `LOCAL_DATASET_ROOT`
  - `LOCAL_MANIFEST_PATH`
  - `DATASET_VERSION`
  - `R2_CONFIG_FILE`
  - `SKIP_R2_PREFLIGHT=1` (if your policy disallows head-bucket)
- Download script:
  - `DOWNLOAD_ROOT`
  - `DATASET_VERSION`
  - `R2_CONFIG_FILE`
  - `SKIP_R2_PREFLIGHT=1`
