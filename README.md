# Spotify Cross-Border Diffusion Predictor

A machine learning system that predicts which countries a newly charting song will spread to on Spotify's Top 200 charts — and roughly when. Built to help music labels allocate promotional budgets with better timing and market focus.

## Executive Summary

When a song first appears on a national Spotify chart, labels face a narrow window to decide where to promote it internationally. This system provides a data-driven answer by scoring all 62 Spotify markets and returning a ranked top-5 list of the most likely next countries, along with estimated days to chart entry.

**Key results (2021 test set):**
- **recall@5 = 0.670** — the model captures two-thirds of actual entry markets in its top-5 predictions
- **ndcg@5 = 0.727** — correct countries are ranked higher in the list
- **hit_rate@5 = 0.878** — at least one correct market appears in the top-5 for 87.8% of spreading tracks
- **Timing MAE = 7.52 days** — directional timing estimates suited for weekly planning cycles

The dominant predictive signal is not what a song sounds like, but the structural relationship between launch and target markets — shared cultural taste infrastructure (proxied through language), artist chart history, and day-0 chart footprint.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Cloudflare R2 Data Bucket](#cloudflare-r2-data-bucket)
- [Data Pipeline](#data-pipeline)
- [Project Structure](#project-structure)
- [Infrastructure & Pipeline Overview](#infrastructure--pipeline-overview)
- [Notebooks Guide](#notebooks-guide)
- [Model Architecture](#model-architecture)
- [Feature Engineering](#feature-engineering)
- [Language Detection via LLM](#language-detection-via-llm)
- [Streamlit Frontend](#streamlit-frontend)
- [Evaluation & Results](#evaluation--results)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Getting Started

### Prerequisites

- Python 3.13+
- AWS CLI (for dataset download): `brew install awscli`

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ML_Group_AB

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download the Datasets

All datasets and model artifacts are hosted on **Cloudflare R2** (S3-compatible storage). Credentials are provided separately via Moodle.

```bash
# 1. Copy the credential template and fill in the values from Moodle
cp scripts/r2.env.example scripts/r2.env
# Edit scripts/r2.env with the provided R2_ENDPOINT, R2_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# 2. Download all dataset versions
bash scripts/download_from_r2.sh                          # v1 (raw, default)
DATASET_VERSION=v2 bash scripts/download_from_r2.sh       # v2 (cleaned)
DATASET_VERSION=v3_features bash scripts/download_from_r2.sh  # v3 (feature-engineered splits)
DATASET_VERSION=v1_aux bash scripts/download_from_r2.sh    # auxiliary reference tables

# 3. Download model artifacts
DATASET_VERSION=artifacts bash scripts/download_from_r2.sh
```

### Run the Streamlit App

```bash
streamlit run app.py
```

This opens the web frontend with two modes:
- **Demo**: Browse pre-computed predictions from the 2021 test set
- **Production**: Input a custom song and get live top-5 country predictions

### Run the Notebooks

The notebooks are numbered and should be run in order:

```bash
jupyter notebook notebooks/
```

| Notebook | Purpose | Creates |
|----------|---------|---------|
| `01_data_downloading` | Download raw data from R2 | `datasets/v1/` |
| `02_EDA` | Exploratory Data Analysis on 26M chart observations | — |
| `03_data_cleaning` | Clean, deduplicate, type-cast (DuckDB) | `datasets/v2/` |
| `04_feature_engineering` | Build 102-feature pair-level matrix | `datasets/v3_features/` |
| `05_model_development_evaluation` | Train & evaluate XGBoost pipeline | `artifacts/` |

The `notebooks/development/` folder contains experimental notebooks (06–11) documenting prototype iterations and hyperparameter tuning.

---

## Cloudflare R2 Data Bucket

We use **Cloudflare R2** (S3-compatible object storage) to host all datasets and model artifacts — anything too large to commit to git. The bucket is organized by dataset version, with each version stored under `dataset/<version>/`.

### Bucket Structure

```
s3://<bucket>/dataset/
├── v1/                    # Raw merged Parquet (~1.5 GB)
│   ├── full/              #   All 46 columns, Hive-partitioned by year
│   │   ├── year=2017/
│   │   ├── year=2018/
│   │   ├── year=2019/
│   │   ├── year=2020/
│   │   └── year=2021/
│   └── slim/              #   Subset of columns (~624 MB)
│       └── year=YYYY/
│
├── v2/                    # Cleaned & deduplicated Parquet (~787 MB)
│   └── full/
│       └── year=YYYY/
│
├── v3_features/           # Feature-engineered splits (~3.8 GB)
│   ├── train.parquet      #   ≤2019, downsampled 5:1 (268K rows)
│   ├── val.parquet        #   2020, natural rate (1.49M rows)
│   ├── test.parquet       #   2021, natural rate (1.37M rows)
│   ├── full.parquet       #   Full non-downsampled dataset
│   ├── full/              #   Hive-partitioned full dataset
│   ├── song_language_cache.json  # LLM-detected song languages
│   └── manifest.json
│
├── v1_aux/                # Auxiliary reference tables
│   ├── countries_reference_clean.parquet
│   ├── cultural_distance_long.parquet
│   └── cultural_distance_top5.parquet
│
└── artifacts/             # Trained models & evaluation results (~263 MB)
    ├── models/
    │   └── xgboost_final_pipeline/
    │       ├── stage2_country_ranker.json
    │       ├── stage3_days_to_entry_regressor.json
    │       └── training_summary.json
    └── evaluations/
        └── xgboost_final_pipeline/
            ├── test_predictions.parquet
            ├── feature_importance.parquet
            └── final_evaluation_plots.png
```

All files are **Parquet** format (columnar, compressed) except for JSON metadata and model files.

### For TAs / Professors

The `.env` file with R2 credentials is submitted separately via **Moodle**. To access the data:

1. Install AWS CLI: `brew install awscli`
2. Place the provided `.env` content into `scripts/r2.env`
3. Download datasets using the commands in [Getting Started](#download-the-datasets)

### Credential Format (`scripts/r2.env`)

```bash
R2_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com"
R2_BUCKET="<bucket-name>"
AWS_ACCESS_KEY_ID="<key>"
AWS_SECRET_ACCESS_KEY="<secret>"
DATASET_VERSION="v1"
AWS_REGION="auto"
```

### Shell Commands Reference

```bash
# Download a specific version
DATASET_VERSION=v2 bash scripts/download_from_r2.sh

# Download to a custom directory
DOWNLOAD_ROOT=./my_data bash scripts/download_from_r2.sh

# Skip preflight check if bucket policy blocks head-bucket
SKIP_R2_PREFLIGHT=1 bash scripts/download_from_r2.sh

# Upload a dataset version (contributors only)
DATASET_VERSION=v2 bash scripts/upload_to_r2.sh
```

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `Unable to locate credentials` | Ensure `scripts/r2.env` exists with all four credential fields |
| `R2 preflight failed` | Verify endpoint, bucket name, and keys; try `SKIP_R2_PREFLIGHT=1` |
| Permission denied | Credentials need at least `ListBucket` + `GetObject` permissions |
| `aws: command not found` | Install AWS CLI: `brew install awscli` |

---

## Data Pipeline

The data flows through three versioned stages, each building on the previous:

### v1 — Raw Merge

- 26.2M observations from 5 years (2017–2021) of daily Spotify chart data across 69 countries + Global
- Raw CSVs merged into yearly Hive-partitioned Parquet files
- Created by: `scripts/process_first_dataset.sh` + `scripts/process_first_dataset_pandas.py`

### v2 — Cleaned & Deduplicated

- Row-level filters: removed Global aggregate, 7 low-coverage countries (South Korea, Luxembourg, Russia, Ukraine, Egypt, Morocco, Saudi Arabia), null `track_id` rows, and 755 exact duplicates
- Type corrections via DuckDB `TRY_CAST` (37 of 46 columns stored as VARCHAR in v1)
- Result: 24.4M rows across 62 markets
- Created by: `notebooks/03_data_cleaning.ipynb`

### v3 — Feature-Engineered

- Each track expanded to up to 62 rows (one per candidate target country)
- 102 features per (track, country) pair after pruning
- Temporal split: **train** (≤2019), **val** (2020), **test** (2021)
- Training set downsampled to 5:1 negative-to-positive ratio; val/test kept at natural rate (~0.7–1%)
- Created by: `notebooks/04_feature_engineering.ipynb`

### Auxiliary Data (`v1_aux`)

- **Country metadata**: population, continent, primary language, government type
- **Cultural distance matrix**: Hofstede 6-dimensional cultural distances between all country pairs
- **Top-5 cultural neighbors**: precomputed nearest neighbors for corridor features
- Source files: `Countries Data By Aadarsh Vani.csv`, `cultural_distance_matrix.csv`
- Processed by: `scripts/prepare_auxiliary_datasets.sh`

---

## Project Structure

```
ML_Group_AB/
│
├── app.py                             # Streamlit entry point
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── src/                               # Core Python source code
│   ├── config.py                      #   Paths, constants, 62 countries, language mappings
│   ├── data.py                        #   Data loading, artist lookup, prediction row builder
│   ├── models.py                      #   Model loading & scoring
│   ├── metrics.py                     #   Ranking & regression evaluation metrics
│   └── pipeline.py                    #   Pipeline helper utilities
│
├── views/                             # Streamlit frontend pages
│   ├── demo.py                        #   Demo mode — browse test set predictions
│   └── production.py                  #   Production mode — custom song predictions
│
├── notebooks/                         # Analysis & modeling notebooks (run in order)
│   ├── 01_data_downloading.ipynb      #   Download datasets from Cloudflare R2
│   ├── 02_EDA.ipynb                   #   Exploratory Data Analysis
│   ├── 03_data_cleaning.ipynb         #   Data cleaning (v1 → v2)
│   ├── 04_feature_engineering.ipynb   #   Feature engineering (v2 → v3)
│   ├── 05_model_development_evaluation.ipynb  # Model training & evaluation
│   └── development/                   #   Experimental notebooks (prototypes, tuning)
│       ├── 06–11 various XGBoost experiments
│
├── scripts/                           # Data pipeline & cloud storage
│   ├── download_from_r2.sh            #   Download datasets from Cloudflare R2
│   ├── upload_to_r2.sh                #   Upload datasets to Cloudflare R2
│   ├── r2.env.example                 #   R2 credentials template
│   ├── process_first_dataset.sh       #   v1 merge orchestration
│   ├── process_first_dataset_pandas.py#   Parquet merging & partitioning
│   ├── prepare_auxiliary_datasets.sh  #   Auxiliary data prep
│   ├── prepare_auxiliary_datasets.py  #   Cultural distance & country metadata
│   └── generate_manifest.py           #   Dataset metadata generation
│
├── datasets/                          # Data storage (versioned, hosted on R2, not in git)
│   ├── v1/                            #   Raw merged Parquet (~1.5 GB)
│   ├── v2/                            #   Cleaned & deduplicated (~787 MB)
│   ├── v3_features/                   #   Feature-engineered splits (~3.8 GB)
│   ├── v1_aux/                        #   Auxiliary reference tables
│   ├── Countries Data By Aadarsh Vani.csv
│   └── cultural_distance_matrix.csv
│
└── artifacts/                         # Trained models & evaluation results (~263 MB)
    ├── models/
    │   └── xgboost_final_pipeline/    #   Production model
    └── evaluations/
        └── xgboost_final_pipeline/    #   Test set metrics, predictions, plots
```

---

## Infrastructure & Pipeline Overview

```
                        DATA PIPELINE
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  Raw Spotify Charts CSVs (5 years, 69 countries)             │
 │           │                                                  │
 │           ▼                                                  │
 │  ┌─────────────────┐    ┌──────────────────────┐             │
 │  │  v1: Raw Merge   │───▶│  Cloudflare R2 Bucket │            │
 │  │  (Parquet, yearly)│◀───│  (S3-compatible)      │            │
 │  └────────┬────────┘    └──────────────────────┘             │
 │           │                                                  │
 │           ▼                                                  │
 │  ┌─────────────────┐   + Auxiliary Data                      │
 │  │  v2: Cleaned     │   (Country metadata, Hofstede           │
 │  │  62 markets      │    cultural distances)                  │
 │  └────────┬────────┘                                         │
 │           │                                                  │
 │           ▼                                                  │
 │  ┌─────────────────┐   + LLM Language Detection              │
 │  │  v3: Feature     │   (song_language_cache.json)            │
 │  │  Engineered      │                                        │
 │  │  102 features    │                                        │
 │  │  per (track,     │                                        │
 │  │   country) pair  │                                        │
 │  └────────┬────────┘                                         │
 │           │                                                  │
 │    train (≤2019) / val (2020) / test (2021)                  │
 └───────────┼──────────────────────────────────────────────────┘
             │
             ▼
          PREDICTION PIPELINE
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │   OBSERVATION POINT (day 0)                                  │
 │   Song first appears on a national Top 200 chart             │
 │   Known: audio features, artist history, chart footprint,    │
 │          market properties, cultural distances                │
 │           │                                                  │
 │           ▼                                                  │
 │   For each of 62 candidate countries:                        │
 │           │                                                  │
 │    ┌──────┴──────────────────────────────────┐               │
 │    │                                         │               │
 │    ▼                                         ▼               │
 │  ┌───────────────────────────┐   ┌───────────────────────────┐
 │  │  Stage 1                   │   │  Stage 2                   │
 │  │  Country Ranker             │   │  Days-to-Entry Regressor   │
 │  │  (XGBRanker, rank:ndcg)    │   │  (XGBRegressor, log1p)     │
 │  │  Ranks all 62 countries     │   │  Predicts timing per       │
 │  │  by entry likelihood        │   │  country (1–60 days)       │
 │  └─────────────┬───────────────┘   └─────────────┬─────────────┘
 │                │                                 │               │
 │                └────────────────┬────────────────┘               │
 │                                 ▼                                │
 │                  ┌───────────────────────────┐                   │
 │                  │  OUTPUT                    │                   │
 │                  │  Top-5 countries + days    │                   │
 │                  │                           │                   │
 │                  │  1. Spain    — ~2 days    │                   │
 │                  │  2. Germany  — ~5 days    │                   │
 │                  │  3. France   — ~8 days    │                   │
 │                  │  4. Italy    — ~12 days   │                   │
 │                  │  5. Colombia — ~18 days   │                   │
 │                  └───────────────────────────┘                   │
 │                                                                  │
 └──────────────────────────────────────────────────────────────────┘
             │
             ▼
          STREAMLIT FRONTEND
 ┌──────────────────────────────────────────────────────────────┐
 │  Demo Mode: Browse test set predictions with metrics         │
 │  Production Mode: Input a song and get live predictions      │
 └──────────────────────────────────────────────────────────────┘
```

---

## Notebooks Guide

### 01 — Data Downloading
Downloads raw datasets from Cloudflare R2 using AWS CLI. Handles credential management through `scripts/r2.env`.

### 02 — Exploratory Data Analysis
Analyzes 26.2M observations across 69 markets (pre-cleaning). Key findings: 80% of tracks never leave their home market, cross-border spread is heavily front-loaded (35% on day 1), and predictable diffusion corridors exist (US→Anglophone, European→Asian markets).

### 03 — Data Cleaning
Transforms v1 into v2. Drops Global aggregate, 7 low-coverage countries, null `track_id` rows, and 755 duplicates. Casts 37 VARCHAR columns to proper types using DuckDB.

### 04 — Feature Engineering
Constructs the (track, target_country) pair-level feature matrix. Expands each track into 62 rows with 102 features in 7 groups. Includes LLM-based song language detection, Hofstede cultural distance imputation, and strict day-0 leakage prevention.

### 05 — Model Development & Evaluation
Trains and evaluates the 2-stage XGBoost pipeline. Includes Logistic Regression and naive popularity baselines, Optuna hyperparameter tuning (50 trials), temporal cross-validation, bootstrap significance testing, feature importance analysis, and SHAP interpretation.

### Development Notebooks (06–11)
Experimental iterations documenting the model development journey: XGBoost classifier prototypes, ranker tuning, will-spread gate ablation, and multitask pipeline experiments.

---

## Model Architecture

The production model is a **2-stage XGBoost pipeline** stored in `artifacts/models/xgboost_final_pipeline/`.

### Stage 1: Country Ranker
- **Type**: XGBRanker with `rank:ndcg` objective
- **Question**: "Which countries are most likely to chart this song?"
- **Approach**: Scores all 62 candidate countries simultaneously, optimized for ndcg@5 via Optuna (50 trials)
- Best CV ndcg@5: 0.948

### Stage 2: Days-to-Entry Regressor
- **Type**: XGBRegressor with log1p target transform
- **Question**: "How many days until the song charts in each country?"
- **Output**: Predicted days to entry, clipped to [1, 60] days
- Test MAE: 7.52 days

### Baselines
- **Naive Popularity**: Ranks countries by average daily streams (no learning)
- **Logistic Regression**: Linear classifier with `class_weight='balanced'`
- **Linear Regression**: Timing prediction baseline (MAE: 9.43 days)

---

## Feature Engineering

102 features organized into 7 groups, computed per (track, country) pair:

| Group | Count | Features |
|-------|-------|----------|
| **Chart Footprint** | 61 | `rank_<country>` for each Spotify market (rank 1–200 or 0 if not charting) |
| **Audio Features** | 12 | Spotify audio features: danceability, energy, valence, tempo, acousticness, speechiness, instrumentalness, liveness, key, loudness, mode, time_signature |
| **Track Metadata** | 5 | duration, explicit flag, days since release, Friday release flag, viral50 presence |
| **Artist History** | 7 | Prior chart count, unique regions, best rank, unique tracks, multi-artist flag, country ratio, prior success in target country |
| **Target Country** | 9 | Population, average daily streams, new entry rate (30-day lookback), 6 continent dummies |
| **Origin–Target Relationship** | 6 | Cultural distance (Hofstede min), cultural distance missing flag, same language flag, song language matches target, same continent, neighbor count already entered |
| **Temporal** | 2 | Observation month and year |

All features use only information available at day 0 to prevent leakage.

---

## Language Detection via LLM

Song language is a critical feature — shared cultural taste infrastructure (proxied through language) is the single strongest predictor, with entry-rate lifts ranging from ~47x for French to ~2.5x for English.

Since Spotify's API does not provide song language, we used a **local LLM** (Qwen 3.5 via Ollama) to infer the primary language from track title and artist name.

- **20 supported languages**: English, Spanish, Portuguese, French, German, Italian, Dutch, Japanese, Korean, Chinese, Hindi, Arabic, Turkish, Polish, Swedish, Romanian, Indonesian, Thai, Tagalog, Vietnamese
- **Results cached** in `datasets/v3_features/song_language_cache.json`
- Two derived features: `same_language_flag` (country-level), `song_lang_matches_target` (song-level)

---

## Streamlit Frontend

The app (`streamlit run app.py`) provides two modes:

### Demo Mode
- Browse pre-computed predictions from the **2021 test set**
- Select any track that spread to 2+ countries
- View the model's top-5 country predictions with correctness indicators
- Expandable sections for feature importance (SHAP) and baseline comparison
- Metrics displayed: recall@5, hit_rate@5, ndcg@5, timing MAE

### Production Mode
- Input a **custom song**: artist name, title, language, release date, explicit flag
- Add the song's **current chart footprint** (which countries it's already in, with ranks)
- Adjust **audio features** via sliders (pre-filled with training medians)
- Get **top-5 predicted countries** with scores and estimated days to chart entry
- View all 62 country scores as a horizontal bar chart
- Artist history is automatically looked up from the v2 dataset via DuckDB

---

## Evaluation & Results

Evaluation uses a **temporal split** to prevent data leakage:
- **Train**: ≤ 2019 | **Validation**: 2020 | **Test**: 2021

### Country Ranking (Test Set)

| Model | recall@5 | ndcg@5 | hit_rate@5 |
|-------|----------|--------|------------|
| Naive Popularity | 0.074 | 0.109 | 0.250 |
| Logistic Regression | 0.601 | 0.622 | 0.824 |
| **XGBRanker** | **0.670** | **0.727** | **0.878** |

XGBRanker improves recall@5 by +11.5% and ndcg@5 by +16.8% over Logistic Regression (p < 0.0001, paired bootstrap).

### Timing Prediction (Test Set)

| Model | MAE | % within 7 days |
|-------|-----|-----------------|
| Linear Regression | 9.43 days | 52.0% |
| **XGBRegressor** | **7.52 days** | **69.7%** |

Full evaluation results, predictions, and plots are stored in `artifacts/evaluations/xgboost_final_pipeline/`.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **ML Framework** | XGBoost (ranker, regressor) |
| **Hyperparameter Tuning** | Optuna (50 trials, temporal CV) |
| **Data Processing** | DuckDB, Pandas, Parquet |
| **Feature Importance** | SHAP |
| **Frontend** | Streamlit |
| **Cloud Storage** | Cloudflare R2 (S3-compatible) |
| **Language Detection** | Qwen 3.5 via Ollama (local LLM) |
| **Evaluation** | scikit-learn, custom ranking metrics, bootstrap CI |
| **Visualization** | Matplotlib, Seaborn |
| **Language** | Python 3.13+ |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
