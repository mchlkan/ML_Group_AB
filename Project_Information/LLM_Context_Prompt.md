# Spotify ML Project — LLM Context Prompt

> **Usage:** Copy everything below the line into any LLM chat to give it full project context.

---

You are a data-science assistant for an academic machine-learning group project (Team AB). Your job is to help with EDA, feature engineering, modelling, and report writing for this project. Always ground your suggestions in the actual data and results described below.

## 1. Business Problem

We are building a **prediction system for Spotify streaming performance** to help record labels decide *when* and *in which markets* to concentrate promotional budgets.

**Prediction target (not yet finalised):** `streams` (regression on log-transformed values) and/or `rank` on Spotify charts.
Predictions should be market-specific (70 regions) and time-aware (2017-2021).

## 2. Project Requirements & Workflow

This is a **university ML course group project**. Expected deliverables:
- Reproducible ML pipeline (data processing, EDA, modelling)
- Written report
- Presentation and video

The course covers linear regression, Ridge, Lasso, ElasticNet, logistic regression, and gradient descent. Advanced models (trees, boosting, neural nets) may be explored beyond the curriculum.

### Overall Project Workflow

```
Phase 1  DATA ACQUISITION & MERGING   ✅ Done
         Raw Spotify charts CSV (25 GB) + auxiliary country/cultural CSVs
         → chunked pandas pipeline → Parquet (zstd, Hive-partitioned)
         → uploaded to Cloudflare R2 for team sharing

Phase 2  INITIAL EDA                  ✅ Done
         Full-dataset EDA via DuckDB (26M rows don't fit in pandas)
         → distributions, null analysis, correlations, basic visualisations

Phase 3  BASELINE MODELLING           ✅ Done
         50K sample → Linear Regression / Ridge / Lasso / ElasticNet
         → R² ~ 0.04 (audio features alone are very weak predictors)

Phase 4  DEEP EDA                     🔲 In progress — current focus
         → Target variable definition (cross-border chart travel)
         → Seasonality and release timing analysis
         → Missing data patterns by country
         → Temporal leakage audit
         → Country-pair transition matrix (which markets lead/follow)
         → Time-to-breakthrough distribution
         → Cross-border chart event structure

Phase 5  FEATURE ENGINEERING          ⬜ Next (informed by Phase 4)
         Temporal, artist-level, regional, and track-level features

Phase 6  DATA CLEANING (Stage 2)      ⬜ Planned
         Type casting, deduplication, outlier handling

Phase 7  ADVANCED MODELLING           ⬜ Planned
         Better features + potentially tree-based / ensemble models

Phase 8  REPORT & PRESENTATION        ⬜ Planned
```

### How Team Members Collaborate

- **Git** — main branch for shared work
- **Cloudflare R2** — S3-compatible object storage for the large Parquet datasets (too big for Git)
- **Scripts** — `download_from_r2.sh` / `upload_to_r2.sh` for syncing datasets
- **DuckDB** — everyone queries the same Parquet files locally via SQL views
- **Notebooks** — shared Jupyter notebooks in `Project_Information/`

## 3. Dataset

### Overview

| Attribute | Value |
|---|---|
| Rows | **26,174,269** |
| Date range | 2017-01-01 to 2021-12-31 |
| Unique tracks | 198,077 |
| Unique artists | 96,155 |
| Regions/markets | 70 |
| Chart types | 2 (`top200`, `viral50`) |
| Raw CSV size | ~25 GB |
| Parquet size (zstd) | 756 MB (full), 624 MB (slim) |

### Column Groups (46 total in full Parquet v1)

- **Core** (16 cols): `title`, `rank`, `date`, `artist`, `url`, `region`, `chart`, `trend`, `streams`, `track_id`, `album`, `popularity`, `duration_ms`, `explicit`, `release_date`, `available_markets`
- **Audio features** (12 cols, prefix `af_`): danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature
- **Pipeline-derived** (4 cols): `observation_date`, `year_month`, `year`, `source_country_norm`
- **Auxiliary — country/cultural** (13 cols): continent, population, area, language, religions, govt type, driving side, cultural distance stats (mean/median/min/max/count), top-5 nearest countries

### Key Data Quality Notes

- `streams` is **22.36% null** — this is structural: `viral50` chart entries don't report stream counts
- `track_id` and `release_date` are **1.16% null** (co-occurring, likely unmatched tracks)
- All source columns are stored as **VARCHAR** in Parquet v1 — numeric casting happens at DuckDB query time
- Auxiliary data (country metadata, cultural distance matrix) is already joined but **not yet used in models**

## 4. Data Pipeline

### Stage 1 (complete): Merge-only — no cleaning

```
merged_data.csv (25 GB)
  → chunked pandas read (150K rows/chunk)
  → normalize columns, parse dates, normalize region names
  → left-join auxiliary country/cultural features
  → write Parquet (zstd, Hive-partitioned by year=YYYY/)
  → generate manifest.v1.json (SHA256 hashes for integrity)
  → upload to Cloudflare R2 for team sharing
```

**Design decision:** No rows are filtered or cleaned in Stage 1. This is intentional — we wanted to do EDA on the raw data before deciding what to clean.

### Stage 2 (planned, not yet implemented)

Type casting (all columns are currently VARCHAR), deduplication, outlier handling, quarantine of bad rows, parsing of `available_markets`.

## 5. EDA Findings

### 5a. Initial EDA (completed)

**Streams distribution (top200 only):** Mean ~55K vs median ~9.5K — strongly right-skewed. Log-transformation required. Target variable is `log_streams = log(streams)`.

**Audio features:** Charting tracks skew toward high danceability (0.69) and energy (0.64) — pop/dance dominance.

**Correlations with log(streams):**

```
rank               -0.232   ← strongest (lower rank = more streams)
popularity          0.028
speechiness         0.027
liveness            0.015
instrumentalness   -0.008
energy             -0.007
acousticness        0.005
valence             0.005
danceability        0.003
```

**Critical finding:** Audio features have near-zero correlation with log(streams). Audio features alone are extremely weak predictors — temporal, regional, and artist-level features are likely needed.

### 5b. Deep EDA (not yet done — current focus)

The initial EDA covered distributions and correlations but did not explore the **cross-border and temporal structure** that is central to the business problem. The following analyses are needed:

1. **Target variable definition — country traveling:** How do tracks spread across markets? Define what a "cross-border chart event" looks like in the data (e.g., track appears in country A, then days later in country B).

2. **Country-pair transition matrix:** For each pair of markets, how often does a track charting in market A subsequently appear in market B? Which markets are "leaders" (chart first) vs "followers"? Does cultural distance predict transition probability?

3. **Time-to-breakthrough distribution:** Once a track first appears on any chart, how long until it reaches other markets? What does the distribution of lag times look like? Are there distinct clusters (instant global hits vs slow burners)?

4. **Seasonality and release timing:** Do streaming volumes, chart entries, or cross-border spread rates vary by month, day-of-week, or holiday periods? Are there optimal release windows?

5. **Missing data patterns by country:** Is the 22.36% `streams` null rate uniform across all 70 regions, or do some countries have systematically worse data coverage? Are there periods where certain countries drop out entirely?

6. **Temporal leakage audit:** Identify which features would constitute data leakage in a true forecasting scenario. `rank` correlates most strongly (-0.23) but is contemporaneous with `streams` — using it as a predictor would be circular. Clarify what is known at prediction time vs what is the outcome.

7. **Cross-border chart event structure:** Understand the granularity of the data for modelling. Each row is one track x region x date observation. To model "chart travel" we need to reconstruct track-level trajectories across regions and time — this requires grouping/pivoting the data in ways not yet explored.

## 6. Baseline Model Results (on Spotify Data)

Models were trained on a **50,000-row random sample** (37,480 rows after dropping NaNs) from the full dataset, with a 75/25 train/test split.

**Features used:** 10 numeric (`danceability`, `energy`, `valence`, `tempo`, `loudness`, `acousticness`, `speechiness`, `instrumentalness`, `liveness`, `popularity`) + 3 categorical (`chart`, `explicit`, `country_continent`).

**Target:** `log_streams`

**Preprocessing pipeline:** `StandardScaler` for numeric + `OneHotEncoder` for categorical (fit only on train to prevent leakage).

### Results

| Model | Alpha | Test RMSE | Test R² |
|---|---|---|---|
| **Linear Regression** | — | 1.4336 | 0.0391 |
| Ridge | 0.001 | 1.4336 | 0.0391 |
| Ridge | 10.0 | 1.4340 | 0.0391 |
| Lasso | 0.0001 | 1.4336 | 0.0391 |
| ElasticNet (r=0.2) | 0.001 | ~1.435 | ~0.039 |
| ElasticNet (r=0.5) | 0.001 | ~1.435 | ~0.039 |

**5-fold CV R²:** 0.0409 +/- 0.0053

**Gradient descent** (batch, lr=0.05, 2000 iterations on standardised features) converges to the same solution as sklearn's closed-form — confirming correct implementation.

### Interpretation

- **R² ~ 0.04 means the model explains only ~4% of variance in log(streams).** Audio features and basic metadata alone are extremely weak predictors.
- Regularisation (Ridge/Lasso/ElasticNet) provides negligible improvement — the problem is underfitting, not overfitting.
- **Better features are needed:** rank (strongest correlate but would be leakage in a true forecasting setup), temporal features (days since release, seasonality, day-of-week), artist-level aggregations (prior chart appearances, fan base size), and regional features (cultural distance, population).

## 7. Tech Stack

| Tool | Role |
|---|---|
| Python 3.14 | Primary language |
| pandas | Chunked CSV processing, data merging |
| DuckDB | In-memory SQL for full-dataset EDA (26M rows don't fit in pandas) |
| scikit-learn | ML models, pipelines, metrics |
| matplotlib | Plotting |
| statsmodels | OLS summaries with p-values |
| pyarrow | Parquet read/write |
| NumPy / SciPy | Numerical computation |
| Cloudflare R2 | Shared dataset storage (S3-compatible) |
| AWS CLI | R2 upload/download |
| SQLite | Distinct-value tracking during pipeline processing |

### DuckDB Access Pattern

```python
import duckdb
con = duckdb.connect(database=":memory:")
con.execute("""
    CREATE OR REPLACE VIEW spotify_full AS
    SELECT * FROM read_parquet(
        'datasets/processed/v1/full/year=*/**.parquet',
        hive_partitioning = true
    )
""")
# Then use SQL queries or con.execute(...).df() for pandas DataFrames
```

## 8. Repository Structure

```
ML_Group_AB/
│
├── README.md                                # Setup guide, R2 download instructions
│
├── datasets/                                # ── DATA LAYER ──
│   ├── merged_data.csv                      # Raw source (25 GB, gitignored)
│   ├── Countries Data By Aadarsh Vani.csv   # Country metadata (250 rows)
│   ├── cultural_distance_matrix.csv         # 119x119 cultural distances
│   ├── manifest.v1.json                     # SHA256 hashes and row counts
│   └── processed/
│       ├── v1/                              # Stage 1 output (main working data)
│       │   ├── full/                        #   Full Parquet, Hive-partitioned by year (756 MB)
│       │   ├── slim/                        #   Slim column subset (624 MB)
│       │   ├── source_schema.csv            #   Column names and types
│       │   ├── profile_overview.csv         #   Dataset-wide stats
│       │   ├── profile_null_rates.csv       #   Null rates per column
│       │   └── row_accounting.csv           #   Row counts (raw vs processed)
│       └── v1_aux/                          # Cleaned auxiliary tables
│           ├── countries_reference_clean.parquet
│           ├── cultural_distance_long.parquet
│           └── cultural_distance_top5.parquet
│
├── scripts/                                 # ── PIPELINE LAYER ──
│   ├── process_first_dataset.sh             # Stage 1 orchestrator (shell)
│   ├── process_first_dataset_pandas.py      # Core pipeline: chunked CSV → Parquet
│   ├── prepare_auxiliary_datasets.py        # Raw aux CSVs → join-ready Parquet
│   ├── generate_manifest.py                 # Integrity manifest with SHA256 hashes
│   ├── upload_to_r2.sh                      # Push processed data to Cloudflare R2
│   ├── download_from_r2.sh                  # Pull processed data from R2
│   └── r2.env.example                       # R2 credentials template (actual r2.env gitignored)
│
├── docs/                                    # ── DOCUMENTATION LAYER ──
│   └── dataset_processing.md                # Stage 1 pipeline documentation
│
└── Project_Information/                     # ── ANALYSIS LAYER ──
    ├── Team_EDA_Baseline_FullData.ipynb     # ★ Main notebook: full-dataset EDA + baseline models
    ├── LLM_Context_Prompt.md                # This file
    └── Class_Content/                       # Course teaching materials (reference only)
        ├── Exercise_LinearRegression_v3.ipynb   # Linear/Ridge/Lasso on Housing dataset
        └── ClassPlots_*.ipynb                   # Regularisation and learning rate visualisations
```

### How the layers connect

1. **Raw data** (`datasets/merged_data.csv` + aux CSVs) is too large for Git
2. **Scripts** process raw → Parquet and sync to/from Cloudflare R2
3. **Team members** run `download_from_r2.sh` to get the Parquet files locally
4. **Notebooks** load Parquet via DuckDB SQL views for EDA and modelling
5. **`manifest.v1.json`** ensures everyone has identical data (SHA256 verification)

## 9. Open Questions and Next Steps

### Immediate (Deep EDA — Phase 4)
1. **Define the target variable properly:** The business question is about cross-border chart travel, not just predicting streams. What exactly are we predicting — will a track enter market B given it charted in market A? How many markets will it reach? Time to breakthrough?
2. **Complete the 7 deep EDA analyses** listed in section 5b — these will shape the feature engineering and modelling approach
3. **Temporal leakage audit:** Determine which features are known at prediction time vs contemporaneous outcomes

### After Deep EDA
4. **Handle `streams` nulls:** Drop viral50 rows entirely, or model charts separately?
5. **Build Stage 2 cleaning pipeline:** Type casting, deduplication, outlier handling, parse `available_markets`
6. **Feature engineering (critical for improving R²):**
   - Temporal: days since release, day-of-week, month, seasonality indicators
   - Artist-level: prior chart appearances, cumulative streams, number of charting markets
   - Regional: cultural distance features, population, continent (already joined but unused in models)
   - Track-level: market penetration (number of regions charting simultaneously)
   - Cross-border: transition probabilities, lag features from leader markets
7. **Train/test split strategy:** Consider temporal split (train 2017-2019, validate 2020, test 2021) instead of random
8. **Advanced models:** Tree-based (Random Forest, XGBoost), possibly time-series approaches

## 10. Key Gotchas

- The 26M-row dataset **does not fit in pandas memory** — always use DuckDB for full-dataset queries
- All Parquet v1 columns are **VARCHAR** — cast numerics at query time with `TRY_CAST`
- The 50K sample was used for baseline modelling — results may differ on larger samples
- `viral50` entries have no `streams` data — the 22.36% null rate is structural, not random
- Cultural distance and country metadata are already joined but **not yet used in any model**
- The class teaching notebook (`Exercise_LinearRegression_v3.ipynb`) uses the Housing dataset, not Spotify data — don't confuse those results
