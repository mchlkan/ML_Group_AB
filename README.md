# ML_Group_AB

We propose a prediction system that helps labels allocate promotional budgets with better timing and market focus.

## Dataset Pipeline

### Stage 1 (current): Merge Only for EDA

Stage 1 creates merged Parquet datasets without cleaning/filtering so the team can do EDA first.

- Partitioning: yearly (`year=YYYY` folders)
- Default chunk size: `150000`
- Goal: produce one large merged dataset for EDA, then clean in Stage 2

Run base merge:

```bash
bash scripts/process_first_dataset.sh
```

Run merge with auxiliary features:

```bash
bash scripts/prepare_auxiliary_datasets.sh
JOIN_AUX=1 bash scripts/process_first_dataset.sh
```

Outputs are written to `datasets/processed/v1/` and a manifest is generated at `datasets/manifest.v1.json` (or `manifest.<DATASET_VERSION>.json` when overridden).

### What gets merged from auxiliary datasets

When `JOIN_AUX=1`, Stage 1 adds:

- Country metadata:
  - `country_continent`, `country_population`, `country_area`
  - `country_official_language`, `country_major_religions`, `country_govt_type`, `country_driving_side`
- Cultural features:
  - `cultural_distance_mean`, `cultural_distance_median`, `cultural_distance_min`, `cultural_distance_max`, `cultural_distance_count`
  - `cultural_top5_targets`

Join key:

- base `region` (normalized) -> aux `source_country_norm`

### Auxiliary dataset prep

```bash
bash scripts/prepare_auxiliary_datasets.sh
```

Creates join-ready tables in `datasets/processed/v1_aux/`.

See full details in `docs/dataset_processing.md`.

## R2 Upload/Download Scripts

One-command helpers are included:

- `scripts/upload_to_r2.sh`
- `scripts/download_from_r2.sh`
- `scripts/r2.env.example`

Setup once:

```bash
cp scripts/r2.env.example scripts/r2.env
# edit scripts/r2.env with your real values
```

Upload processed dataset + manifest:

```bash
bash scripts/upload_to_r2.sh
```

Download dataset version:

```bash
bash scripts/download_from_r2.sh
```

Teammate flow:

1. Install AWS CLI (`brew install awscli`).
2. Copy your `scripts/r2.env` values (or create their own read-only `scripts/r2.env`).
3. Run `bash scripts/download_from_r2.sh`.

Notes:

- Upload/download scripts auto-load and export variables from `scripts/r2.env`.
- For non-default versions, set `DATASET_VERSION` and keep matching paths:

```bash
DATASET_VERSION=v2 bash scripts/upload_to_r2.sh
```

## Teammate Guide: Download Dataset from Cloudflare R2 (Read-only)

Use this if you only need to pull data for analysis/modeling.

### 1) Get read-only R2 credentials

Ask the maintainer for read-only credentials scoped to the dataset bucket:

- `R2_ENDPOINT`
- `R2_BUCKET`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Recommended permissions for teammates are read-only (`ListBucket` + `GetObject`).

### 2) Install prerequisites

```bash
brew install awscli
```

### 3) Create local config

From repo root:

```bash
cp scripts/r2.env.example scripts/r2.env
```

Edit `scripts/r2.env` and fill values:

```bash
R2_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com"
R2_BUCKET="<bucket-name>""
AWS_ACCESS_KEY_ID="<read-only-key>""
AWS_SECRET_ACCESS_KEY="<read-only-secret>""
DATASET_VERSION="v1"
```

### 4) Download dataset

```bash
bash scripts/download_from_r2.sh
```

Default local output:

- `datasets/<DATASET_VERSION>/`

### 5) Useful overrides

Download a different version:

```bash
DATASET_VERSION=v2 bash scripts/download_from_r2.sh
```

Download into a custom local folder:

```bash
DOWNLOAD_ROOT=./datasets_from_r2/v1 bash scripts/download_from_r2.sh
```

### 6) Troubleshooting

- `Unable to locate credentials`:
  - Ensure `scripts/r2.env` exists and has all four credential fields.
- `R2 preflight failed`:
  - Verify endpoint, bucket name, and key/secret.
  - If your bucket policy blocks `head-bucket`, run:

```bash
SKIP_R2_PREFLIGHT=1 bash scripts/download_from_r2.sh
```

- Permission denied during download:
  - Ask for read access to the bucket objects (`List` + `GetObject`).
