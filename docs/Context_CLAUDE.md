# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University ML project predicting Spotify cross-border music diffusion across 66 markets (2017–2021). Core task: **Top-5 Country Ranker** — given a song's first chart day, predict which 5 countries it will chart in next (within 60 days). Dataset: 25.1M rows, Hive-partitioned Parquet.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: DuckDB 1.4.4, pandas 3.0.1, scikit-learn 1.8.0, matplotlib 3.10.8.

## Data Pipeline

```
Raw CSV (25 GB) → scripts/process_first_dataset.sh → v1 Parquet (Hive by year)
                → notebooks/03_data_cleaning.ipynb  → v2 Parquet (cleaned, 25.1M rows)
                → notebooks/05_feature_engineering_v2.ipynb → v3 features (train/val/test splits)
```

- **v1** (`datasets/processed/v1/full/`): Raw ingestion, all VARCHAR columns, 46 cols
- **v1_aux** (`datasets/processed/v1_aux/`): Cultural distance matrices, country metadata
- **v2** (`datasets/processed/v2/full/`): Cleaned + typed, 45 cols, 66 countries (dropped Global, South Korea, Russia, Ukraine)
- **v3** (`datasets/processed/v3_features/`): Feature-engineered train/val/test parquet splits

Datasets are gitignored. Team syncs via Cloudflare R2 (`scripts/upload_to_r2.sh` / `scripts/download_from_r2.sh`). R2 credentials go in `scripts/r2.env` (see `r2.env.example`).

## Architecture

**DuckDB is required** for any queries on the full dataset (26M+ rows exceed pandas memory). Pattern:
```python
con = duckdb.connect("path.duckdb")  # disk-backed for memory safety
con.execute("CREATE VIEW v2 AS SELECT * FROM read_parquet('datasets/processed/v2/full/year=*/data_0.parquet', hive_partitioning=true)")
```

**Notebook pipeline** (run sequentially):
- `03_data_cleaning.ipynb` — v1 → v2: type casting, deduplication, region filtering
- `04_feature_engineering.ipynb` — Feature spec (read-only reference, not executable)
- `05_feature_engineering_v2.ipynb` — Executable feature engineering: builds ~102 features per (track, target_country) pair, exports train/val/test parquet

**Feature matrix structure**: One row per `(track_id, target_country)`. Observation point = first chart day. Prediction horizon = 60 days. Rank columns use top200 only; viral50 is a separate binary flag. Temporal split: train ≤2019, val 2020, test 2021 (by track's first chart date).

**Key data columns**:
- `chart`: `'top200'` or `'viral50'` — rank columns and labels use top200 only
- `streams`: 21.5% NULL (viral50 entries have no stream count)
- `release_date`: 1.5% NULL (year-only dates fail DATE cast)
- All v1 source columns are VARCHAR; v2 has proper types

## Important Conventions

- Branch `main` is the default PR target. Feature work happens on topic branches.
- Parquet files use zstd compression and Hive partitioning by year.
- Cultural distance data has 19.4% missing (non-Western countries not in Hofstede database) — impute with median, flag missingness.
