# Spotify Diffusion Predictor

A machine learning system that predicts which countries a song will spread to on Spotify's Top-200 charts and how quickly it will get there. Built to help music labels allocate promotional budgets with better timing and market focus.

Given a song that just entered a national Spotify chart, the system predicts:
- **Which** of the remaining 62 Spotify markets the song is most likely to chart in within 60 days
- **How many days** until each target country's chart entry
- A ranked **top-5** list of the most promising markets

---

## Table of Contents

- [Infrastructure & Pipeline Overview](#infrastructure--pipeline-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Pipeline](#data-pipeline)
- [Cloudflare R2 Data Bucket](#cloudflare-r2-data-bucket)
- [Notebooks Guide](#notebooks-guide)
- [Model Architecture](#model-architecture)
- [Feature Engineering](#feature-engineering)
- [Language Detection via LLM](#language-detection-via-llm)
- [Streamlit Frontend](#streamlit-frontend)
- [Evaluation & Results](#evaluation--results)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Infrastructure & Pipeline Overview

The end-to-end system flows from raw Spotify chart data through three processing stages into a 2-stage XGBoost prediction pipeline, served via a Streamlit frontend.

```
                        DATA PIPELINE
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  Raw Spotify Charts CSVs (5 years, 62 countries)             │
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
 │  │  & Deduplicated  │    cultural distances)                  │
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
 │   OBSERVATION POINT                                          │
 │   Song is charting in countries {X, Y, Z}                    │
 │   Known: audio features, artist history, chart footprint,    │
 │          market properties, cultural distances                │
 │           │                                                  │
 │           ▼                                                  │
 │   For each of 62 remaining candidate countries:              │
 │           │                                                  │
 │    ┌──────┴──────────────────────────────────┐               │
 │    │                                         │               │
 │    ▼                                         ▼               │
 │  ┌───────────────────────────┐   ┌───────────────────────────┐
 │  │  Stage 1                   │   │  Stage 2                   │
 │  │  Country Ranker             │   │  Days-to-Entry Regressor   │
 │  │  (XGBoost LambdaMART)       │   │  (XGBoost Regressor)       │
 │  │  Ranks all 62 countries     │   │  Predicts timing per       │
 │  │  by spread likelihood        │   │  country (1-60 days)       │
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
├── datasets/                          # Data storage (versioned, not in git)
│   ├── v1/                            #   Raw merged Parquet (yearly partitions)
│   ├── v2/                            #   Cleaned & deduplicated Parquet
│   ├── v3_features/                   #   Feature-engineered splits (train/val/test)
│   │   └── song_language_cache.json   #   LLM-detected song languages
│   ├── v1_aux/                        #   Auxiliary reference tables
│   ├── Countries Data By Aadarsh Vani.csv  # Country metadata
│   └── cultural_distance_matrix.csv        # Hofstede cultural distances
│
└── artifacts/                         # Trained models & evaluation results
    ├── models/
    │   └── xgboost_final_pipeline/    #   Production 3-stage model
    │       ├── stage2_country_ranker.json
    │       ├── stage3_days_to_entry_regressor.json
    │       └── training_summary.json
    └── evaluations/
        └── xgboost_final_pipeline/    #   Test set metrics, predictions, plots
            ├── test_predictions.parquet
            ├── feature_importance.parquet
            ├── model_comparison.parquet
            └── final_evaluation_plots.png
```

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

The datasets are stored in a **Cloudflare R2** bucket (S3-compatible). Credentials are provided separately via Moodle (see [Cloudflare R2 Data Bucket](#cloudflare-r2-data-bucket)).

```bash
# 1. Copy the credential template
cp scripts/r2.env.example scripts/r2.env

# 2. Fill in the credentials from the .env file submitted on Moodle
#    (R2_ENDPOINT, R2_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

# 3. Download all dataset versions (v1, v2, v3_features, v1_aux, artifacts, etc.)
bash scripts/download_from_r2.sh
DATASET_VERSION=v2 bash scripts/download_from_r2.sh
DATASET_VERSION=v3_features bash scripts/download_from_r2.sh
DATASET_VERSION=v1_aux bash scripts/download_from_r2.sh
```

### Run the Streamlit App

```bash
streamlit run app.py
```

This opens the web frontend with two modes:
- **Demo**: Browse pre-computed predictions from the 2021 test set
- **Production**: Input a custom song and get live top-5 country predictions

### Run the Notebooks

The notebooks are numbered and should be read/run in order:

```bash
jupyter notebook notebooks/
```

| Notebook | Purpose |
|----------|---------|
| `01_data_downloading` | Download raw data from R2 |
| `02_EDA` | Exploratory Data Analysis on 5 years of chart data |
| `03_data_cleaning` | Clean and deduplicate (v1 → v2) |
| `04_feature_engineering` | Build 102-feature matrix (v2 → v3) |
| `05_model_development_evaluation` | Train & evaluate the 2-stage pipeline |

The `notebooks/development/` folder contains experimental notebooks (06–11) documenting prototype iterations and hyperparameter tuning with Optuna.

---

## Data Pipeline

The data flows through three versioned stages:

### v1 — Raw Merge
- 5 years of daily Spotify Top-200 chart data across 62 countries
- Merged into yearly Parquet partitions (`year=YYYY/`)
- Created by: `scripts/process_first_dataset.sh`

### v2 — Cleaned & Deduplicated
- Row-level filters, type corrections, deduplication
- 7 low-coverage markets excluded (South Korea, Russia, Ukraine, Luxembourg, Egypt, Morocco, Saudi Arabia)
- Created by: `notebooks/03_data_cleaning.ipynb` (uses DuckDB for on-disk processing)

### v3 — Feature-Engineered
- Each track is expanded to 62 rows (one per candidate country)
- 102 pruned features per (track, country) pair
- Temporal split: **train** (≤2019), **val** (2020), **test** (2021)
- Created by: `notebooks/04_feature_engineering.ipynb`

### Auxiliary Data
- **Country metadata**: population, continent, primary language (from `Countries Data By Aadarsh Vani.csv`)
- **Cultural distance matrix**: Hofstede cultural dimensions between all country pairs
- Stored in `datasets/v1_aux/`

---

## Cloudflare R2 Data Bucket

We use **Cloudflare R2** (S3-compatible object storage) to host all datasets and model artifacts. The R2 bucket contains every dataset version (v1, v2, v3_features, v1_aux) as well as trained model artifacts — anything too large to commit to git.

### For TAs / Professors

The `.env` file with R2 credentials is submitted separately via **Moodle**. To access the data:

1. Install AWS CLI: `brew install awscli`
2. Place the provided `.env` content into `scripts/r2.env`
3. Download all datasets:

```bash
bash scripts/download_from_r2.sh
DATASET_VERSION=v2 bash scripts/download_from_r2.sh
DATASET_VERSION=v3_features bash scripts/download_from_r2.sh
DATASET_VERSION=v1_aux bash scripts/download_from_r2.sh
```

### Credential Format (`scripts/r2.env`)

```bash
R2_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com"
R2_BUCKET="<bucket-name>"
AWS_ACCESS_KEY_ID="<key>"
AWS_SECRET_ACCESS_KEY="<secret>"
DATASET_VERSION="v1"
AWS_REGION="auto"
```

### Available Dataset Versions on R2

| Version | Contents |
|---------|----------|
| `v1` (default) | Raw merged Parquet (yearly partitions) |
| `v2` | Cleaned & deduplicated Parquet |
| `v3_features` | Feature-engineered train/val/test splits |
| `v1_aux` | Auxiliary reference tables (country metadata, cultural distances) |

### Useful Overrides

```bash
# Download a specific version
DATASET_VERSION=v2 bash scripts/download_from_r2.sh

# Download to a custom directory
DOWNLOAD_ROOT=./my_data bash scripts/download_from_r2.sh

# Skip preflight check if bucket policy blocks head-bucket
SKIP_R2_PREFLIGHT=1 bash scripts/download_from_r2.sh
```

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `Unable to locate credentials` | Ensure `scripts/r2.env` exists with all four credential fields |
| `R2 preflight failed` | Verify endpoint, bucket name, and keys; try `SKIP_R2_PREFLIGHT=1` |
| Permission denied | Credentials need at least `ListBucket` + `GetObject` permissions |

---

## Notebooks Guide

### 01 — Data Downloading
Downloads raw datasets from Cloudflare R2 using AWS CLI. Handles credential management through `scripts/r2.env`.

### 02 — Exploratory Data Analysis
Analyzes 5 years of Spotify daily chart data across 62 markets. Investigates how songs spread across borders and identifies features that predict cross-border diffusion.

### 03 — Data Cleaning
Transforms v1 into a clean v2 dataset. Applies row-level filters, type corrections, and deduplication using DuckDB for efficient on-disk processing.

### 04 — Feature Engineering
Constructs the (track, target_country) pair-level feature matrix. Transforms one row per track into 62 rows (one per country) with 102 features organized in 7 groups (see [Feature Engineering](#feature-engineering)).

### 05 — Model Development & Evaluation
Trains and evaluates the final 3-stage XGBoost pipeline. Includes baseline comparisons, temporal cross-validation, hyperparameter tuning with Optuna, and comprehensive evaluation metrics.

### Development Notebooks (06–11)
Experimental iterations documenting the model development journey: XGBoost prototypes, ranker tuning, classifier development, and multitask pipeline experiments.

---

## Model Architecture

The production model is a **2-stage XGBoost pipeline** stored in `artifacts/models/xgboost_final_pipeline/`.

### Stage 1: Country Ranker
- **Type**: XGBoost Ranker (LambdaMART listwise ranking)
- **Question**: "Which countries are most likely to chart this song?"
- **Approach**: Scores all 62 candidate countries simultaneously using listwise ranking loss
- Optimized directly for NDCG@5 via Optuna (34 trials)
- CV NDCG@5: 0.951

### Stage 2: Days-to-Entry Regressor
- **Type**: XGBoost Regressor with log1p target transform
- **Question**: "How many days until the song charts in each country?"
- **Output**: Predicted days to entry, clipped to [1, 60] days
- CV MAE: 8.48 days

### Baselines
- **Naive Popularity**: Ranks countries by average daily streams (no learning)
- **Logistic Regression**: Simple linear classifier baseline
- **Linear Regression**: Timing prediction baseline

---

## Feature Engineering

102 features organized into 7 groups, computed per (track, country) pair:

| Group | Count | Features |
|-------|-------|----------|
| **Chart Footprint** | 62 | `rank_<country>` for each of 62 Spotify markets (current rank 1–200 or 0) |
| **Audio Features** | 12 | Spotify audio features: danceability, energy, valence, tempo, acousticness, speechiness, instrumentalness, liveness, key, loudness, mode, time_signature |
| **Track Metadata** | 5 | duration, explicit flag, days since release, Friday release flag, viral 50 presence |
| **Artist History** | 7 | Prior chart count, unique regions, best rank, unique tracks, multi-artist flag, country ratio, prior success in target country |
| **Target Country Priors** | 3 | Population, average daily streams, new entry rate (last 30 days) |
| **Origin–Target Relationship** | 5 | Language match, song language matches target, same continent, cultural distance (Hofstede), neighbor count already entered |
| **Temporal** | 2 | Observation month and year |

Two zero-gain features were pruned during training (104 → 102).

---

## Language Detection via LLM

Song languages are a critical feature for predicting cross-border diffusion (e.g., Spanish songs spreading to Latin American markets). Since Spotify's API does not provide song language metadata, we used an **LLM-based approach** to determine the primary language of each track.

- **20 supported languages**: English, Spanish, Portuguese, French, German, Italian, Dutch, Japanese, Korean, Chinese, Hindi, Arabic, Turkish, Polish, Swedish, Romanian, Indonesian, Thai, Tagalog, Vietnamese
- **Results cached** in `datasets/v3_features/song_language_cache.json` to avoid repeated API calls
- Language features used: `same_language_flag`, `song_lang_matches_target` (does the song's language match the target country's primary language)

---

## Streamlit Frontend

The app (`streamlit run app.py`) provides two modes:

### Demo Mode
- Browse pre-computed predictions from the **2021 test set**
- Select any track that spread to 2+ countries
- View the model's top-5 country predictions with correctness indicators
- Expandable sections for feature importance (SHAP) and baseline comparison
- Metrics displayed: Recall@5, Hit Rate@5, NDCG@5, Timing MAE

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

### Ranking Metrics
| Metric | Description |
|--------|-------------|
| **Recall@5** | Fraction of actual spread countries captured in top-5 |
| **Hit Rate@5** | Does the top-5 contain at least one correct country? |
| **NDCG@5** | Ranking quality (rewards correct predictions higher in the list) |
| **MAP@5** | Mean Average Precision across all cutoffs |

### Timing Metrics
| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error in predicted days to entry |
| **Pct within 3/7 days** | Fraction of predictions within N-day error threshold |

Full evaluation results, predictions, feature importances, and comparison plots are stored in `artifacts/evaluations/xgboost_final_pipeline/`.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **ML Framework** | XGBoost (classifier, ranker, regressor) |
| **Hyperparameter Tuning** | Optuna |
| **Data Processing** | DuckDB, Pandas, Parquet |
| **Feature Importance** | SHAP |
| **Frontend** | Streamlit |
| **Cloud Storage** | Cloudflare R2 (S3-compatible) |
| **Language Detection** | LLM-based classification |
| **Evaluation** | scikit-learn, custom ranking metrics |
| **Visualization** | Matplotlib, Seaborn |
| **Language** | Python 3.13+ |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
